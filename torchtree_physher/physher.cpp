#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include "phycpp/physher.hpp"
namespace py = pybind11;

using double_np =
    py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

PYBIND11_MODULE(physher, m) {
  py::class_<TreeLikelihoodInterface>(m, "TreeLikelihoodModel")
      .def(py::init<const std::vector<std::pair<std::string, std::string>> &,
                    TreeModelInterface *, SubstitutionModelInterface *,
                    SiteModelInterface *, std::optional<BranchModelInterface *>,
                    bool, bool, bool>())
      .def("log_likelihood", &TreeLikelihoodInterface::LogLikelihood)
      .def("gradient",
           [](TreeLikelihoodInterface &self) {
             double_np gradient = double_np(self.gradientLength_);
             self.Gradient(gradient.mutable_data());
             return gradient;
           })
      .def("request_gradient", &TreeLikelihoodInterface::RequestGradient);

  py::class_<TreeModelInterface>(m, "TreeModelInterface");
  py::class_<UnRootedTreeModelInterface, TreeModelInterface>(
      m, "UnRootedTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &>())
      .def("set_parameters",
           [](UnRootedTreeModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](UnRootedTreeModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<ReparameterizedTimeTreeModelInterface, TreeModelInterface>(
      m, "ReparameterizedTimeTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &,
                    const std::vector<double>>())
      .def("set_parameters",
           [](ReparameterizedTimeTreeModelInterface &self,
              double_np parameters) { self.SetParameters(parameters.data()); })
      .def("parameters",
           [](ReparameterizedTimeTreeModelInterface &self) {
             double_np parameters = double_np(self.parameterCount_);
             self.GetParameters(parameters.mutable_data());
             return parameters;
           })
      .def("get_node_heights",
           [](ReparameterizedTimeTreeModelInterface &self) {
             double_np parameters = double_np(self.parameterCount_);
             self.GetNodeHeights(parameters.mutable_data());
             return parameters;
           })
      .def("gradient_transform_jvp",
           [](ReparameterizedTimeTreeModelInterface &self,
              double_np height_gradient) {
             double_np gradient = double_np(self.parameterCount_);
             self.GradientTransformJVP(gradient.mutable_data(),
                                       height_gradient.data());
             return gradient;
           })
      .def("gradient_transform_jvp",
           [](ReparameterizedTimeTreeModelInterface &self,
              double_np height_gradient, double_np heights) {
             double_np gradient = double_np(self.parameterCount_);
             self.GradientTransformJVP(gradient.mutable_data(),
                                       height_gradient.data(), heights.data());
             return gradient;
           })
      .def("gradient_transform_jacobian",
           [](ReparameterizedTimeTreeModelInterface &self) {
             double_np gradient = double_np(self.parameterCount_);
             self.GradientTransformJacobian(gradient.mutable_data());
             return gradient;
           })
      .def("transform_jacobian",
           &ReparameterizedTimeTreeModelInterface::TransformJacobian);

  py::class_<SubstitutionModelInterface>(m, "SubstitutionModelInterface");

  py::class_<JC69Interface, SubstitutionModelInterface>(m, "JC69")
      .def(py::init<>())
      .def("set_parameters",
           [](JC69Interface &self, double_np parameters) {
             return self.SetParameters(parameters.data());
           })
      .def("parameters", [](JC69Interface &self) {
        double_np empty;
        return empty;
      });

  py::class_<HKYInterface, SubstitutionModelInterface>(m, "HKY")
      .def(py::init<double, const std::vector<double> &>())
      .def("set_kappa", &HKYInterface::SetKappa)
      .def("set_frequencies",
           [](HKYInterface &self, double_np parameters) {
             return self.SetFrequencies(parameters.data());
           })
      .def("set_parameters",
           [](HKYInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](HKYInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<GTRInterface, SubstitutionModelInterface>(m, "GTR")
      .def(py::init<const std::vector<double> &, const std::vector<double> &>())
      .def("set_rates",
           [](GTRInterface &self, double_np parameters) {
             return self.SetRates(parameters.data());
           })
      .def("set_frequencies",
           [](GTRInterface &self, double_np parameters) {
             return self.SetFrequencies(parameters.data());
           })
      .def("set_parameters",
           [](GTRInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](GTRInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<SiteModelInterface>(m, "SiteModelInterface")
      .def("set_mu", &ConstantSiteModelInterface::SetMu);
  py::class_<ConstantSiteModelInterface, SiteModelInterface>(
      m, "ConstantSiteModel")
      .def(py::init<std::optional<double>>())
      .def("set_parameters",
           [](ConstantSiteModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](ConstantSiteModelInterface &self) {
        if (self.parameterCount_ == 0) {
          double_np empty;
          return empty;
        } else {
          double_np parameters = double_np(self.parameterCount_);
          self.GetParameters(parameters.mutable_data());
          return parameters;
        }
      });

  py::class_<InvariantSiteModelInterface, SiteModelInterface>(
      m, "InvariantSiteModel")
      .def(py::init<double, std::optional<double>>())
      .def("set_proportion_invariant",
           &InvariantSiteModelInterface::SetProportionInvariant)
      .def("set_parameters",
           [](InvariantSiteModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](InvariantSiteModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<DiscretizedSiteModelInterface, SiteModelInterface>(
      m, "DiscretizedConstantSiteModel")
      .def("set_proportion_invariant",
           &DiscretizedSiteModelInterface::SetProportionInvariant)
      .def("set_parameters",
           [](DiscretizedSiteModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](DiscretizedSiteModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<WeibullSiteModelInterface, DiscretizedSiteModelInterface>(
      m, "WeibullSiteModel")
      .def(py::init<double, size_t, std::optional<double>,
                    std::optional<double>>())
      .def("set_shape", &WeibullSiteModelInterface::SetShape);

  py::class_<GammaSiteModelInterface, DiscretizedSiteModelInterface>(
      m, "GammaSiteModel")
      .def(py::init<double, size_t, std::optional<double>,
                    std::optional<double>>())
      .def("set_shape", &GammaSiteModelInterface::SetShape)
      .def("set_epsilon", &GammaSiteModelInterface::SetEpsilon);

  py::class_<BranchModelInterface>(m, "BranchModelInterface")
      .def("parameters", [](BranchModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });
  py::class_<StrictClockModelInterface, BranchModelInterface>(
      m, "StrictClockModel")
      .def(py::init<double, TreeModelInterface *>())
      .def("set_parameters",
           [](StrictClockModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("set_rate", &StrictClockModelInterface::SetRate);

  py::class_<SimpleClockModelInterface, BranchModelInterface>(
      m, "SimpleClockModel")
      .def(py::init<const std::vector<double> &, TreeModelInterface *>())
      .def("set_parameters",
           [](SimpleClockModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           });

  py::class_<CoalescentModelInterface>(m, "CoalescentModelInterface")
      .def("log_likelihood", &CoalescentModelInterface::LogLikelihood)
      .def("gradient",
           [](CoalescentModelInterface &self) {
             double_np parameters = double_np(self.gradientLength_);
             self.Gradient(parameters.mutable_data());
             return parameters;
           })
      .def("request_gradient", &CoalescentModelInterface::RequestGradient)
      .def("set_parameters",
           [](CoalescentModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](CoalescentModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<ConstantCoalescentModelInterface, CoalescentModelInterface>(
      m, "ConstantCoalescentModel")
      .def(py::init<double, TreeModelInterface *>());

  py::class_<PiecewiseConstantCoalescentInterface, CoalescentModelInterface>(
      m, "PiecewiseConstantCoalescentModel")
      .def(py::init<const std::vector<double>, TreeModelInterface *>());

  py::class_<PiecewiseConstantCoalescentGridInterface,
             CoalescentModelInterface>(m,
                                       "PiecewiseConstantCoalescentGridModel")
      .def(py::init<const std::vector<double>, TreeModelInterface *, double>());

  py::class_<CTMCScaleModelInterface>(m, "CTMCScaleModel")
      .def(py::init<const std::vector<double>, TreeModelInterface *>())
      .def("log_likelihood", &CTMCScaleModelInterface::LogLikelihood)
      .def("gradient",
           [](CTMCScaleModelInterface &self) {
             double_np gradient = double_np(self.gradientLength_);
             self.Gradient(gradient.mutable_data());
             return gradient;
           })
      .def("request_gradient", &CTMCScaleModelInterface::RequestGradient)
      .def("set_parameters",
           [](CTMCScaleModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](CTMCScaleModelInterface &self) {
        double_np empty;
        return empty;
      });

  py::module gradient_flags = m.def_submodule("gradient_flags");
  py::enum_<GradientFlags>(gradient_flags, "coalescent_gradient_flags")
      .value("TREE_RATIO", GradientFlags::TREE_RATIO,
             "gradient of reparameterizated node heights")
      .value("TREE_HEIGHT", GradientFlags::TREE_HEIGHT,
             "gradient of node heights")
      .value("THETA", GradientFlags::COALESCENT_THETA,
             "gradient of population size parameters")
      .export_values();

  py::module tree_likelihood_gradient_flags =
      m.def_submodule("tree_likelihood_gradient_flags");
  py::enum_<TreeLikelihoodGradientFlags>(tree_likelihood_gradient_flags,
                                         "tree_likelihood_gradient_flags")
      .value("TREE_HEIGHT", TreeLikelihoodGradientFlags::TREE_HEIGHT,
             "gradient of reparameterizated node heights")
      .value("SITE_MODEL", TreeLikelihoodGradientFlags::SITE_MODEL,
             "gradient of site model parameters")
      .value("SUBSTITUTION_MODEL",
             TreeLikelihoodGradientFlags::SUBSTITUTION_MODEL,
             "gradient of substitution model parameters")
      .value("BRANCH_MODEL", TreeLikelihoodGradientFlags::BRANCH_MODEL,
             "gradient of branch model parameters")
      .export_values();
}
