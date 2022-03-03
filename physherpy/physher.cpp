#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

namespace py = pybind11;

extern "C" {
#include "phyc/parameters.h"
#include "phyc/sequence.h"
#include "phyc/simplex.h"
#include "phyc/treelikelihood.h"
#include "phyc/treetransform.h"
}

using double_np =
    py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

class Interface {
 public:
  virtual void SetParameters(double_np parameters) = 0;
  virtual double_np GetParameters() = 0;

  Model *model_;
};

class TreeModelInterface : public Interface {
 public:
  TreeModelInterface(const std::string &newick,
                     const std::vector<std::string> &taxa,
                     std::optional<const std::vector<double>> dates) {
    char **taxa_p = new char *[taxa.size()];
    for (size_t i = 0; i < taxa.size(); i++) {
      taxa_p[i] = const_cast<char *>(taxa[i].c_str());
    }
    model_ = new_TreeModel_from_newick(
        newick.c_str(), taxa_p, dates.has_value() ? dates->data() : NULL);
    delete[] taxa_p;

    treeModel_ = reinterpret_cast<Tree *>(model_->obj);
    nodeCount_ = Tree_node_count(treeModel_);
    tipCount_ = Tree_tip_count(treeModel_);

    InitializeMap(taxa);
  }
  virtual ~TreeModelInterface() {}

  void InitializeMap(const std::vector<std::string> &taxa) {
    nodeMap_.resize(nodeCount_);
    for (size_t i = 0; i < nodeCount_; i++) {
      Node *node = Tree_node(treeModel_, i);
      if (Node_isleaf(node)) {
        std::string str(Node_name(node));
        auto it = find(taxa.begin(), taxa.end(), str);
        if (it != taxa.end()) {
          nodeMap_[node->id] = it - taxa.begin();
        } else {
          std::cerr << "Could not find taxon " << str << " in taxon list"
                    << std::endl;
        }
      } else {
        nodeMap_[node->id] = node->class_id + tipCount_;
      }
    }
  }

  size_t nodeCount_;
  size_t tipCount_;
  std::vector<size_t> nodeMap_;
  Tree *treeModel_;
};

class UnRootedTreeModelInterface : public TreeModelInterface {
 public:
  UnRootedTreeModelInterface(const std::string &newick,
                             const std::vector<std::string> &taxa)
      : TreeModelInterface(newick, taxa, std::nullopt) {}

  virtual ~UnRootedTreeModelInterface() {}

  void SetParameters(double_np parameters) override {
    auto data = parameters.data();
    Node **nodes = Tree_nodes(treeModel_);
    Node *root = Tree_root(treeModel_);
    for (size_t i = 0; i < nodeCount_; i++) {
      Node *node = nodes[i];
      if (node != root && root->right != node) {
        Node_set_distance(node, data[nodeMap_[node->id]]);
      }
    }
  }
  double_np GetParameters() override { return {}; }
};

class ReparameterizedTimeTreeModelInterface : public TreeModelInterface {
 public:
  ReparameterizedTimeTreeModelInterface(const std::string &newick,
                                        const std::vector<std::string> &taxa,
                                        const std::vector<double> dates)
      : TreeModelInterface(newick, taxa, dates) {
    transformModel_ = reinterpret_cast<Model *>(model_->data);
  }

  virtual ~ReparameterizedTimeTreeModelInterface() {}

  void SetParameters(double_np parameters) override {
    auto data = parameters.data();
    TreeTransform *tt = reinterpret_cast<TreeTransform *>(transformModel_->obj);
    Parameters_set_values(tt->parameters, data);
  }

  double_np GetParameters() override { return {}; }

  double_np GetNodeHeights() {
    double_np heights(tipCount_ - 1);
    auto data = heights.mutable_data();
    Tree_update_heights(treeModel_);
    Node **nodes = Tree_nodes(treeModel_);
    for (size_t i = 0; i < nodeCount_; i++) {
      if (!Node_isleaf(nodes[i])) {
        data[nodes[i]->class_id] = Node_height(nodes[i]);
      }
    }
    return heights;
  }
  double_np GradientTransformJVP(double_np height_gradient) {
    double_np gradient(tipCount_ - 1);
    auto height_gradient_data = height_gradient.data();
    auto gradient_data = gradient.mutable_data();
    Tree_node_transform_jvp(treeModel_, height_gradient_data, gradient_data);
    return gradient;
  }

  double_np GradientTransformJacobian() {
    double_np gradient(tipCount_ - 1);
    auto gradient_data = gradient.mutable_data();
    Tree_node_transform_jacobian_gradient(treeModel_, gradient_data);
    return gradient;
  }

  double TransformJacobian() {
    TreeTransform *tt = reinterpret_cast<TreeTransform *>(transformModel_->obj);
    return tt->log_jacobian(tt);
  }

 protected:
  Model *transformModel_;
};

class BranchModelInterface : public Interface {
 public:
  virtual ~BranchModelInterface() {}

  void SetParameters(double_np parameters) override {
    Parameters_set_values(branchModel_->rates, parameters.data());
  }
  double_np GetParameters() override {
    double_np values = double_np(Parameters_count(branchModel_->rates));
    auto data = values.mutable_data();
    for (size_t i = 0; i < Parameters_count(branchModel_->rates); i++) {
      data[i] = Parameters_value(branchModel_->rates, i);
    }
    return values;
  }
  void SetRates(double_np rates) {
    Parameters_set_values(branchModel_->rates, rates.data());
  }

 protected:
  BranchModel *branchModel_;
};

class StrictClockModelInterface : public BranchModelInterface {
 public:
  StrictClockModelInterface(double rate, const TreeModelInterface &treeModel) {
    Parameter *p = new_Parameter("", rate, NULL);
    p->model = MODEL_BRANCHMODEL;
    branchModel_ = new_StrictClock_with_parameter(treeModel.treeModel_, p);
    free_Parameter(p);
    model_ = new_BranchModel2("", branchModel_, treeModel.model_, NULL);
  }
  virtual ~StrictClockModelInterface() {}

  void SetRate(double rate) {
    Parameters_set_value(branchModel_->rates, 0, rate);
  }
};

class SubstitutionModelInterface : public Interface {
 public:
  virtual ~SubstitutionModelInterface() {}

 protected:
  Model *Initialize(const std::string &name, Parameters *rates,
                    Model *frequencies) {
    DataType *datatype = new_NucleotideDataType();
    substModel_ = SubstitutionModel_factory(
        name.c_str(), datatype, reinterpret_cast<Simplex *>(frequencies->obj),
        NULL, rates, NULL);
    Model *model = new_SubstitutionModel2("", substModel_, frequencies, NULL);
    free_DataType(datatype);
    return model;
  }

  SubstitutionModel *substModel_;
};

class JC69Interface : public SubstitutionModelInterface {
 public:
  JC69Interface() {
    Simplex *frequencies_simplex = new_Simplex("", 4);
    Model *frequencies_model = new_SimplexModel("", frequencies_simplex);
    model_ = Initialize("JC69", NULL, frequencies_model);
    frequencies_model->free(frequencies_model);
  }

  virtual ~JC69Interface() {}

  void SetParameters(double_np parameters) override {}
  double_np GetParameters() override { return {}; }
};

class HKYInterface : public SubstitutionModelInterface {
 public:
  HKYInterface(double kappa, const std::vector<double> &frequencies) {
    Parameters *kappa_parameters = new_Parameters(1);
    Parameters_move(kappa_parameters, new_Parameter("", kappa, NULL));
    Simplex *frequencies_simplex =
        new_Simplex_with_values("", frequencies.data(), frequencies.size());
    Model *frequencies_model = new_SimplexModel("id", frequencies_simplex);
    model_ = Initialize("hky", kappa_parameters, frequencies_model);
    free_Parameters(kappa_parameters);
    frequencies_model->free(frequencies_model);
  }
  virtual ~HKYInterface() {}
  void SetKappa(double kappa) {
    Parameters_set_value(substModel_->rates, 0, kappa);
  }
  void SetFrequencies(double_np frequencies) {
    substModel_->simplex->set_values(substModel_->simplex, frequencies.data());
  }
  void SetParameters(double_np parameters) override {
    auto data = parameters.data();
    SetKappa(data[1]);
    substModel_->simplex->set_values(substModel_->simplex, data + 1);
  }
  double_np GetParameters() override { return {}; }
};

class GTRInterface : public SubstitutionModelInterface {
 public:
  GTRInterface(const std::vector<double> &rates,
               const std::vector<double> &frequencies) {
    Parameters *rates_parameters = NULL;
    // simplex
    if (rates.size() == 6) {
      Simplex *rates_simplex =
          new_Simplex_with_values("", rates.data(), rates.size());
    } else {
      rates_parameters = new_Parameters(5);
      for (auto rate : rates) {
        Parameters_move(rates_parameters, new_Parameter("", rate, NULL));
      }
    }
    Simplex *frequencies_simplex =
        new_Simplex_with_values("", frequencies.data(), frequencies.size());
    Model *frequencies_model = new_SimplexModel("id", frequencies_simplex);
    model_ = Initialize("gtr", rates_parameters, frequencies_model);
    free_Parameters(rates_parameters);
    frequencies_model->free(frequencies_model);
  }
  virtual ~GTRInterface() {}
  void SetRates(double_np rates) {
    Parameters_set_values(substModel_->rates, rates.data());
  }
  void SetFrequencies(double_np frequencies) {
    substModel_->simplex->set_values(substModel_->simplex, frequencies.data());
  }
  void SetParameters(double_np parameters) override {
    auto data = parameters.data();
    Parameters_set_values(substModel_->rates, data);
    substModel_->simplex->set_values(substModel_->simplex, data + 3);
  }
  double_np GetParameters() override { return {}; }
};

class SiteModelInterface : public Interface {
 public:
  virtual ~SiteModelInterface() {}

 protected:
  SiteModel *siteModel_;
};

class ConstantSiteModelInterface : public SiteModelInterface {
 public:
  explicit ConstantSiteModelInterface(std::optional<double> mu) {
    siteModel_ = new_SiteModel_with_parameters(
        NULL, NULL, 1, DISTRIBUTION_UNIFORM, false, QUADRATURE_QUANTILE_MEDIAN);
    if (mu.has_value()) {
      siteModel_->mu = new_Parameter("mu", *mu, NULL);
      siteModel_->mu->model = MODEL_SITEMODEL;
    }
    model_ = new_SiteModel2("sitemodel", siteModel_, NULL);
  }

  virtual ~ConstantSiteModelInterface() {
    // siteModel_->free(siteModel_);
  }
  void SetMu(double mu) { Parameter_set_value(siteModel_->mu, mu); }
  void SetParameters(double_np parameters) override {
    auto data = parameters.data();
    if (siteModel_->mu != NULL) {
      Parameter_set_value(siteModel_->mu, data[0]);
    }
  }
  double_np GetParameters() override {
    double_np values = double_np(1);
    auto data = values.mutable_data();
    if (siteModel_->mu != NULL) {
      data[0] = Parameter_value(siteModel_->mu);
    }
    return values;
  }
};

class WeibullSiteModelInterface : public SiteModelInterface {
 public:
  WeibullSiteModelInterface(double shape, size_t categories,
                            std::optional<double> mu) {
    Parameter *shape_parameter = new_Parameter("", shape, NULL);
    Parameters *params = new_Parameters(1);
    Parameters_move(params, shape_parameter);
    siteModel_ = new_SiteModel_with_parameters(params, NULL, categories,
                                               DISTRIBUTION_WEIBULL, false,
                                               QUADRATURE_QUANTILE_MEDIAN);
    if (mu.has_value()) {
      siteModel_->mu = new_Parameter("mu", *mu, NULL);
      siteModel_->mu->model = MODEL_SITEMODEL;
    }
    model_ = new_SiteModel2("sitemodel", siteModel_, NULL);
    free_Parameters(params);
  }

  virtual ~WeibullSiteModelInterface() {}

  void SetShape(double shape) {
    Parameters_set_value(siteModel_->rates, 0, shape);
  }
  void SetMu(double mu) { Parameter_set_value(siteModel_->mu, mu); }
  void SetParameters(double_np parameters) override {
    size_t shift = 0;
    auto data = parameters.data();
    if (Parameters_count(siteModel_->rates) > 0) {
      Parameters_set_value(siteModel_->rates, 0, data[0]);
      shift++;
    }
    if (siteModel_->mu != NULL) {
      Parameter_set_value(siteModel_->mu, data[shift]);
      shift++;
    }
  }
  double_np GetParameters() override {
    size_t param_count =
        Parameters_count(siteModel_->rates) + siteModel_->mu != NULL;
    double_np values = double_np(param_count);
    auto data = values.mutable_data();
    size_t shift = 0;
    if (Parameters_count(siteModel_->rates) > 0) {
      data[0] = Parameters_value(siteModel_->rates, 0);
      shift++;
    }
    if (siteModel_->mu != NULL) {
      data[shift] = Parameter_value(siteModel_->mu);
      shift++;
    }
    return values;
  }
};

class TreeLikelihoodInterface {
 public:
  TreeLikelihoodInterface(
      const std::vector<std::pair<std::string, std::string>> &alignment,
      TreeModelInterface *treeModel,
      SubstitutionModelInterface *substitutionModel,
      SiteModelInterface *siteModel,
      std::optional<BranchModelInterface *> branchModel,
      bool use_ambiguities = false)
      : treeModel_(treeModel),
        substitutionModel_(substitutionModel),
        siteModel_(siteModel) {
    branchModel_ = branchModel.has_value() ? *branchModel : NULL;
    Sequences *sequences = new_Sequences(alignment.size());
    for (const auto &sequence : alignment) {
      Sequences_add(sequences, new_Sequence(sequence.first.c_str(),
                                            sequence.second.c_str()));
    }
    sequences->datatype = new_NucleotideDataType();
    SitePattern *sitePattern = new_SitePattern(sequences);
    free_Sequences(sequences);
    Model *mbm = branchModel.has_value() ? branchModel_->model_ : NULL;
    BranchModel *bm = branchModel.has_value()
                          ? reinterpret_cast<BranchModel *>(mbm->obj)
                          : NULL;
    SingleTreeLikelihood *tlk = new_SingleTreeLikelihood(
        reinterpret_cast<Tree *>(treeModel_->model_->obj),
        reinterpret_cast<SubstitutionModel *>(substitutionModel_->model_->obj),
        reinterpret_cast<SiteModel *>(siteModel_->model_->obj), sitePattern, bm,
        !use_ambiguities);
    model_ = new_TreeLikelihoodModel("id", tlk, treeModel_->model_,
                                     substitutionModel_->model_,
                                     siteModel_->model_, mbm);
    RequestGradient();
  }

  double LogLikelihood() { return model_->logP(model_); }

  void RequestGradient(int flags = 0) {
    gradientLength_ = TreeLikelihood_initialize_gradient(model_, flags);
  }

  double_np Gradient() {
    double *gradient = TreeLikelihood_gradient(model_);
    double_np values(
        (branchModel_ == NULL ? gradientLength_ - 2 : gradientLength_));
    auto data = values.mutable_data();
    size_t i = 0;
    size_t j = 0;
    if (branchModel_ == NULL) {
      for (; i < treeModel_->nodeCount_ - 2; i++) {
        data[treeModel_->nodeMap_[i]] = gradient[i];
      }
      i += 2;
      j = treeModel_->nodeCount_ - 2;
    }
    for (; i < gradientLength_; i++, j++) {
      data[j] = gradient[i];
    }
    return values;
  }
  TreeModelInterface *treeModel_;
  SubstitutionModelInterface *substitutionModel_;
  SiteModelInterface *siteModel_;
  BranchModelInterface *branchModel_;
  SitePattern *sitePattern_;
  size_t gradientLength_;

 private:
  Model *model_;
};

PYBIND11_MODULE(physher, m) {
  py::class_<TreeLikelihoodInterface>(m, "TreeLikelihoodModel")
      .def(py::init<const std::vector<std::pair<std::string, std::string>> &,
                    TreeModelInterface *, SubstitutionModelInterface *,
                    SiteModelInterface *, std::optional<BranchModelInterface *>,
                    bool>())
      .def("log_likelihood", &TreeLikelihoodInterface::LogLikelihood)
      .def("gradient", &TreeLikelihoodInterface::Gradient)
      .def("request_gradient", &TreeLikelihoodInterface::RequestGradient);

  py::class_<TreeModelInterface>(m, "TreeModelInterface");
  py::class_<UnRootedTreeModelInterface, TreeModelInterface>(
      m, "UnRootedTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &>())
      .def("set_parameters", &UnRootedTreeModelInterface::SetParameters)
      .def("parameters", &UnRootedTreeModelInterface::GetParameters);

  py::class_<ReparameterizedTimeTreeModelInterface, TreeModelInterface>(
      m, "ReparameterizedTimeTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &,
                    const std::vector<double>>())
      .def("set_parameters",
           &ReparameterizedTimeTreeModelInterface::SetParameters)
      .def("parameters", &ReparameterizedTimeTreeModelInterface::GetParameters)
      .def("get_node_heights",
           &ReparameterizedTimeTreeModelInterface::GetNodeHeights)
      .def("gradient_transform_jvp",
           &ReparameterizedTimeTreeModelInterface::GradientTransformJVP)
      .def("gradient_transform_jacobian",
           &ReparameterizedTimeTreeModelInterface::GradientTransformJacobian)
      .def("transform_jacobian",
           &ReparameterizedTimeTreeModelInterface::TransformJacobian);

  py::class_<SubstitutionModelInterface>(m, "SubstitutionModelInterface");

  py::class_<JC69Interface, SubstitutionModelInterface>(m, "JC69")
      .def(py::init<>())
      .def("set_parameters", &JC69Interface::SetParameters)
      .def("parameters", &JC69Interface::GetParameters);

  py::class_<HKYInterface>(m, "HKY")
      .def(py::init<double, const std::vector<double> &>())
      .def("set_kappa", &HKYInterface::SetKappa)
      .def("set_frequencies", &HKYInterface::SetFrequencies)
      .def("set_parameters", &HKYInterface::SetParameters)
      .def("parameters", &HKYInterface::GetParameters);

  py::class_<GTRInterface>(m, "GTR")
      .def(py::init<const std::vector<double> &, const std::vector<double> &>())
      .def("set_rates", &GTRInterface::SetRates)
      .def("set_frequencies", &GTRInterface::SetFrequencies)
      .def("set_parameters", &GTRInterface::SetParameters)
      .def("parameters", &GTRInterface::GetParameters);

  py::class_<SiteModelInterface>(m, "SiteModelInterface");
  py::class_<ConstantSiteModelInterface, SiteModelInterface>(
      m, "ConstantSiteModel")
      .def(py::init<std::optional<double>>())
      .def("set_mu", &ConstantSiteModelInterface::SetMu)
      .def("set_parameters", &ConstantSiteModelInterface::SetParameters)
      .def("parameters", &ConstantSiteModelInterface::GetParameters);

  py::class_<WeibullSiteModelInterface, SiteModelInterface>(m,
                                                            "WeibullSiteModel")
      .def(py::init<double, size_t, std::optional<double>>())
      .def("set_shape", &WeibullSiteModelInterface::SetShape)
      .def("set_mu", &WeibullSiteModelInterface::SetMu)
      .def("set_parameters", &WeibullSiteModelInterface::SetParameters)
      .def("parameters", &WeibullSiteModelInterface::GetParameters);

  py::class_<BranchModelInterface>(m, "BranchModelInterface");
  py::class_<StrictClockModelInterface, BranchModelInterface>(
      m, "StrictClockModel")
      .def(py::init<double, TreeModelInterface &>())
      .def("set_parameters", &StrictClockModelInterface::SetParameters)
      .def("parameters", &StrictClockModelInterface::GetParameters)
      .def("set_rate", &StrictClockModelInterface::SetRate);
}
