// Copyright 2021-2023 Mathieu Fourment.
// torchtree-physher is free software under the GPLv3; see LICENSE file for
// details.

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
  py::class_<ModelInterface>(m, "ModelInterface")
      .def("set_parameters",
           [](ModelInterface &self, double_np parameters) {
             self.SetParameters(parameters.data());
           })
      .def("parameters", [](ModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetParameters(parameters.mutable_data());
        return parameters;
      });

  py::class_<CallableModelInterface, ModelInterface>(m,
                                                     "CallableModelInterface")
      .def("log_likelihood", &CallableModelInterface::LogLikelihood)
      .def("gradient", [](CallableModelInterface &self) {
        double_np gradient = double_np(self.gradientLength_);
        self.Gradient(gradient.mutable_data());
        return gradient;
      });

  py::class_<TreeLikelihoodInterface, CallableModelInterface>(
      m, "TreeLikelihoodModel")
      .def(py::init<const std::vector<std::pair<std::string, std::string>> &,
                    TreeModelInterface *, SubstitutionModelInterface *,
                    SiteModelInterface *, std::optional<BranchModelInterface *>,
                    bool, bool, bool>())
      .def(py::init<const std::vector<std::string> &,
                    const std::vector<std::string> &, TreeModelInterface *,
                    SubstitutionModelInterface *, SiteModelInterface *,
                    std::optional<BranchModelInterface *>, bool, bool, bool>())
      .def("request_gradient", &TreeLikelihoodInterface::RequestGradient)
      .def("enable_sse", &TreeLikelihoodInterface::EnableSSE);

  py::class_<TreeModelInterface, ModelInterface>(m, "TreeModelInterface");

  py::class_<UnRootedTreeModelInterface, TreeModelInterface>(
      m, "UnRootedTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &>());

  py::class_<TimeTreeModelInterface, TreeModelInterface>(m, "TimeTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &,
                    const std::vector<double>>())
      .def("node_heights", [](TimeTreeModelInterface &self) {
        double_np parameters = double_np(self.parameterCount_);
        self.GetNodeHeights(parameters.mutable_data());
        return parameters;
      });

  py::class_<ReparameterizedTimeTreeModelInterface, TimeTreeModelInterface>(
      m, "ReparameterizedTimeTreeModel")
      .def(py::init<const std::string &, const std::vector<std::string> &,
                    const std::vector<double>, TreeTransformFlags>())
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

  py::class_<SubstitutionModelInterface, ModelInterface>(
      m, "SubstitutionModelInterface")
      .def("set_epsilon", [](SubstitutionModelInterface &self, double epsilon) {
        self.SetEpsilon(epsilon);
      });

  py::class_<JC69Interface, SubstitutionModelInterface>(m, "JC69")
      .def(py::init<>())
      .def("set_parameters",
           [](JC69Interface &self, double_np parameters) {
             throw std::runtime_error(
                 "set_parameters should not be used on a JC69 substitution "
                 "model");
           })
      .def("parameters", [](JC69Interface &self) {
        double_np empty;
        return empty;
      });

  py::class_<HKYInterface, SubstitutionModelInterface>(m, "HKY")
      .def(py::init<double, const std::vector<double> &>())
      .def("set_kappa", &HKYInterface::SetKappa)
      .def("set_frequencies", [](HKYInterface &self, double_np parameters) {
        self.SetFrequencies(parameters.data());
      });

  py::class_<GTRInterface, SubstitutionModelInterface>(m, "GTR")
      .def(py::init<const std::vector<double> &, const std::vector<double> &>())
      .def("set_rates",
           [](GTRInterface &self, double_np parameters) {
             self.SetRates(parameters.data());
           })
      .def("set_frequencies", [](GTRInterface &self, double_np parameters) {
        self.SetFrequencies(parameters.data());
      });

  py::class_<GeneralSubstitutionModelInterface, SubstitutionModelInterface>(
      m, "GeneralSubstitutionModel")
      .def(py::init<DataTypeInterface *, const std::vector<double> &,
                    const std::vector<double> &, const std::vector<unsigned> &,
                    bool>())
      .def("set_rates",
           [](GeneralSubstitutionModelInterface &self, double_np parameters) {
             self.SetRates(parameters.data());
           })
      .def("set_frequencies",
           [](GeneralSubstitutionModelInterface &self, double_np parameters) {
             self.SetFrequencies(parameters.data());
           });

  py::class_<DataTypeInterface>(m, "DataTypeInterface");
  py::class_<GeneralDataTypeInterface, DataTypeInterface>(m, "GeneralDataType")
      .def(py::init<const std::vector<std::string> &,
                    std::optional<const std::map<std::string,
                                                 std::vector<std::string>>>>());
  py::class_<NucleotideDataTypeInterface, DataTypeInterface>(
      m, "NucleotideDataType")
      .def(py::init<>());

  py::class_<SiteModelInterface, ModelInterface>(m, "SiteModelInterface")
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
           &InvariantSiteModelInterface::SetProportionInvariant);

  py::class_<DiscretizedSiteModelInterface, SiteModelInterface>(
      m, "DiscretizedConstantSiteModel")
      .def("set_proportion_invariant",
           &DiscretizedSiteModelInterface::SetProportionInvariant)
      .def("rates",
           [](DiscretizedSiteModelInterface &self) {
             double_np rates = double_np(self.GetCategoryCount());
             self.GetParameters(rates.mutable_data());
             return rates;
           })
      .def("proportions", [](DiscretizedSiteModelInterface &self) {
        double_np proportions = double_np(self.GetCategoryCount());
        self.GetParameters(proportions.mutable_data());
        return proportions;
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

  py::class_<BranchModelInterface, ModelInterface>(m, "BranchModelInterface");

  py::class_<StrictClockModelInterface, BranchModelInterface>(
      m, "StrictClockModel")
      .def(py::init<double, TreeModelInterface *>())
      .def("set_rate", &StrictClockModelInterface::SetRate);

  py::class_<SimpleClockModelInterface, BranchModelInterface>(
      m, "SimpleClockModel")
      .def(py::init<const std::vector<double> &, TreeModelInterface *>())
      .def("set_rates",
           [](SimpleClockModelInterface &self, double_np parameters) {
             self.SetRates(parameters.data());
           });

  py::class_<CoalescentModelInterface, CallableModelInterface>(
      m, "CoalescentModelInterface")
      .def("request_gradient", &CoalescentModelInterface::RequestGradient);

  py::class_<ConstantCoalescentModelInterface, CoalescentModelInterface>(
      m, "ConstantCoalescentModel")
      .def(py::init<double, TimeTreeModelInterface *>());

  py::class_<PiecewiseConstantCoalescentInterface, CoalescentModelInterface>(
      m, "PiecewiseConstantCoalescentModel")
      .def(py::init<const std::vector<double>, TimeTreeModelInterface *>());

  py::class_<PiecewiseConstantCoalescentGridInterface,
             CoalescentModelInterface>(m,
                                       "PiecewiseConstantCoalescentGridModel")
      .def(py::init<const std::vector<double>, TimeTreeModelInterface *, double>());

  py::class_<PiecewiseLinearCoalescentGridInterface, CoalescentModelInterface>(
      m, "PiecewiseLinearCoalescentGridModel")
      .def(py::init<const std::vector<double>, TimeTreeModelInterface *, double>());

  py::class_<CTMCScaleModelInterface, CallableModelInterface>(m,
                                                              "CTMCScaleModel")
      .def(py::init<const std::vector<double>, TreeModelInterface *>())
      .def("request_gradient", &CTMCScaleModelInterface::RequestGradient);

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
             "gradient of all parameters in substitution model")
      .value("SUBSTITUTION_MODEL_RATES",
             TreeLikelihoodGradientFlags::SUBSTITUTION_MODEL_RATES,
             "gradient of rate parameters in substitution model")
      .value("SUBSTITUTION_MODEL_FREQUENCIES",
             TreeLikelihoodGradientFlags::SUBSTITUTION_MODEL_FREQUENCIES,
             "gradient of frequcencies parameters in substitution model")
      .value("BRANCH_MODEL", TreeLikelihoodGradientFlags::BRANCH_MODEL,
             "gradient of branch model parameters")
      .export_values();

  py::module tree_transform_flags = m.def_submodule("tree_transform_flags");
  py::enum_<TreeTransformFlags>(tree_transform_flags, "tree_transform_flags")
      .value("RATIO", TreeTransformFlags::RATIO,
             "tree transform using ratio parameterization")
      .value("SHIFT", TreeTransformFlags::SHIFT,
             "tree transform using shift parameterization")
      .export_values();
}
