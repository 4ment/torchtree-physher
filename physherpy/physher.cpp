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

class Interface {
 public:
  virtual void SetParameters(const std::vector<double> &parameters) = 0;
  virtual std::vector<double> GetParameters() = 0;

  Model *model_;
};

class TreeModelInterface : public Interface {
 public:
  TreeModelInterface(const std::string &newick,
                     const std::vector<std::string> &taxa,
                     std::optional<std::vector<double>> dates) {
    model_ = new_TreeModel_from_newick(
        newick.c_str(), dates.has_value() ? dates->data() : NULL);
    transformModel_ = reinterpret_cast<Model *>(model_->data);
    treeModel_ = reinterpret_cast<Tree *>(model_->obj);
    nodeCount_ = Tree_node_count(treeModel_);
    tipCount_ = Tree_tip_count(treeModel_);
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

  virtual ~TreeModelInterface() {}

  void SetParameters(const std::vector<double> &parameters) override {
    if (transformModel_ != NULL) {
      TreeTransform *tt =
          reinterpret_cast<TreeTransform *>(transformModel_->obj);
      Parameters *p = tt->parameters;
      for (size_t i = 0; i < Parameters_count(p); i++) {
        Parameters_set_value(p, i, parameters[i]);
      }
    } else {
      Node **nodes = Tree_nodes(treeModel_);
      Node *root = Tree_root(treeModel_);
      for (size_t i = 0; i < nodeCount_; i++) {
        Node *node = nodes[i];
        if (node != root && root->right != node) {
          Node_set_distance(node, parameters[nodeMap_[node->id]]);
        }
      }
    }
  }

  std::vector<double> GetParameters() override {
    std::vector<double> values;
    return values;
  }

  size_t nodeCount_;
  size_t tipCount_;
  std::vector<size_t> nodeMap_;
  Tree *treeModel_;

 private:
  Model *transformModel_;
};

class BranchModelInterface : public Interface {
 public:
  virtual ~BranchModelInterface() {}

  void SetParameters(const std::vector<double> &parameters) override {
    for (size_t i = 0; i < parameters.size(); i++) {
      Parameters_set_value(branchModel_->rates, i, parameters[i]);
    }
  }
  std::vector<double> GetParameters() override {
    std::vector<double> values(Parameters_count(branchModel_->rates));
    for (size_t i = 0; i < Parameters_count(branchModel_->rates); i++) {
      values[i] = Parameters_value(branchModel_->rates, i);
    }
    return values;
  }

 protected:
  BranchModel *branchModel_;
};

class StrictClockModelInterface : public BranchModelInterface {
 public:
  StrictClockModelInterface(double rate, TreeModelInterface &treeModel) {
    Parameter *p = new_Parameter("", rate, NULL);
    p->model = MODEL_BRANCHMODEL;
    branchModel_ = new_StrictClock_with_parameter(treeModel.treeModel_, p);
    free_Parameter(p);
    model_ = new_BranchModel2("", branchModel_, treeModel.model_, NULL);
  }
  virtual ~StrictClockModelInterface() {}
};

class SubstitutionModelInterface : public Interface {
 public:
  virtual ~SubstitutionModelInterface() {}

 protected:
  Model *initialize(const std::string &name, Parameters *rates,
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
    model_ = initialize("JC69", NULL, frequencies_model);
    frequencies_model->free(frequencies_model);
  }

  virtual ~JC69Interface() {}

  void SetParameters(const std::vector<double> &parameters) override {}
  std::vector<double> GetParameters() override { return {}; }
};

class HKYInterface : public SubstitutionModelInterface {
 public:
  HKYInterface(double kappa, const std::vector<double> &frequencies) {
    Parameters *kappa_parameters = new_Parameters(1);
    Parameters_move(kappa_parameters, new_Parameter("", kappa, NULL));
    Simplex *frequencies_simplex =
        new_Simplex_with_values("", frequencies.data(), frequencies.size());
    Model *frequencies_model = new_SimplexModel("id", frequencies_simplex);
    model_ = initialize("hky", kappa_parameters, frequencies_model);
    free_Parameters(kappa_parameters);
    frequencies_model->free(frequencies_model);
  }
  virtual ~HKYInterface() {}
  void SetKappa(double kappa) {
    Parameters_set_value(substModel_->rates, 0, kappa);
  }
  void SetFrequencies(std::vector<double> frequencies) {
    substModel_->simplex->set_values(substModel_->simplex, frequencies.data());
  }
  void SetParameters(const std::vector<double> &parameters) override {}
  std::vector<double> GetParameters() override {
    std::vector<double> values;
    return values;
  }
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
    model_ = initialize("gtr", rates_parameters, frequencies_model);
    free_Parameters(rates_parameters);
    frequencies_model->free(frequencies_model);
  }
  virtual ~GTRInterface() {}
  void SetRates(const std::vector<double> &rates) {
    //    Parameters_set_value(substModel_->rates, 0, kappa);
  }
  void SetFrequencies(std::vector<double> frequencies) {
    substModel_->simplex->set_values(substModel_->simplex, frequencies.data());
  }
  void SetParameters(const std::vector<double> &parameters) override {}
  std::vector<double> GetParameters() override {
    std::vector<double> values;
    return values;
  }
};

class SiteModelInterface : public Interface {
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
  void SetParameters(const std::vector<double> &parameters) override {
    if (siteModel_->mu != NULL) {
      Parameter_set_value(siteModel_->mu, parameters[0]);
    }
  }
  std::vector<double> GetParameters() override {
    std::vector<double> values;
    if (siteModel_->mu != NULL) {
      values.push_back(Parameter_value(siteModel_->mu));
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

  virtual ~WeibullSiteModelInterface() {
    // siteModel_->free(siteModel_);
  }

  void SetShape(double shape) {
    Parameters_set_value(siteModel_->rates, 0, shape);
  }
  void SetMu(double mu) { Parameter_set_value(siteModel_->mu, mu); }
  void SetParameters(const std::vector<double> &parameters) override {
    size_t shift = 0;
    if (Parameters_count(siteModel_->rates) > 0) {
      Parameters_set_value(siteModel_->rates, 0, parameters[0]);
      shift++;
    }
    if (siteModel_->mu != NULL) {
      Parameter_set_value(siteModel_->mu, parameters[shift]);
      shift++;
    }
  }
  std::vector<double> GetParameters() override {
    std::vector<double> values;
    size_t shift = 0;
    if (Parameters_count(siteModel_->rates) > 0) {
      values.push_back(Parameters_value(siteModel_->rates, 0));
      shift++;
    }
    if (siteModel_->mu != NULL) {
      values.push_back(Parameter_value(siteModel_->mu));
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
    RequestGradient(0);
  }

  double LogLikelihood() { return model_->logP(model_); }

  void RequestGradient(int flags) {
    gradient_length_ = TreeLikelihood_initialize_gradient(model_, flags);
  }

  std::vector<double> Gradient() {
    double *gradient = TreeLikelihood_gradient(model_);
    std::vector<double> values(
        (branchModel_ == NULL ? gradient_length_ - 2 : gradient_length_));
    size_t i = 0;
    size_t j = 0;
    if (branchModel_ == NULL) {
      for (; i < treeModel_->nodeCount_ - 2; i++) {
        values[treeModel_->nodeMap_[i]] = gradient[i];
      }
      i += 2;
      j = treeModel_->nodeCount_ - 2;
    }
    for (; i < gradient_length_; i++, j++) {
      values[j] = gradient[i];
    }
    return values;
  }
  TreeModelInterface *treeModel_;
  SubstitutionModelInterface *substitutionModel_;
  SiteModelInterface *siteModel_;
  BranchModelInterface *branchModel_;
  SitePattern *sitePattern_;
  size_t gradient_length_;

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

  py::class_<TreeModelInterface>(m, "TreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &,
                    std::optional<std::vector<double>>>())
      .def("set_parameters", &TreeModelInterface::SetParameters)
      .def("parameters", &TreeModelInterface::GetParameters);

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
      .def("parameters", &StrictClockModelInterface::GetParameters);
}
