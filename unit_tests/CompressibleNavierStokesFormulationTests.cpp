//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CompressibleNavierStokesFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/21/18.
//
//

#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "CompressibleNavierStokesFormulationRefactor.hpp"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  static const double TEST_RE = 1e2;
  static const double TEST_PR = 0.713;
  static const double TEST_CV = 1.000;
  static const double TEST_GAMMA = 1.4;
  static const double TEST_CP = TEST_GAMMA * TEST_CV;
  static const double TEST_R = TEST_CP - TEST_CV;
  
  void testForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T,
                      FunctionPtr fc, FunctionPtr fm, FunctionPtr fe,
                      int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 1;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = TEST_RE; // Reynolds number
    
    bool useConformingTraces = true;
    auto form = CompressibleNavierStokesFormulationRefactor::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                               meshTopo, polyOrder, delta_k);
    
    // sanity checks that the constructor has set things up the way we assume:
    TEST_FLOATING_EQUALITY(form.Pr(),    TEST_PR,     tol);
    TEST_FLOATING_EQUALITY(form.Cv(),    TEST_CV,     tol);
    TEST_FLOATING_EQUALITY(form.gamma(), TEST_GAMMA,  tol);
    TEST_FLOATING_EQUALITY(form.Cp(),    TEST_CP,     tol);
    TEST_FLOATING_EQUALITY(form.R(),     TEST_R,      tol);
    TEST_FLOATING_EQUALITY(form.mu(),    1.0/TEST_RE, tol);
    
    BFPtr bf = form.bf();
    RHSPtr rhs = form.rhs();
    
    auto soln = form.solution();
    auto solnIncrement = form.solutionIncrement();
    auto mesh = soln->mesh();

    auto exact_fc = form.exactSolution_fc(u, rho, T);
    auto exact_fe = form.exactSolution_fe(u, rho, T);
    auto exact_fm = form.exactSolution_fm(u, rho, T);
    
    double fc_err = (fc - exact_fc)->l2norm(mesh, cubatureEnrichment);
    if (fc_err > tol)
    {
      success = false;
      out << "FAILURE: fc_err " << fc_err << " > tol " << tol << endl;
      out << "fc expected: " << fc->displayString() << endl;
      out << "fc actual:   " << exact_fc->displayString() << endl;
    }
    
    // here, we use the fact that we're operating in 1D:
    double fm_err = (fm - exact_fm[0])->l2norm(mesh);
    if (fm_err > tol)
    {
      success = false;
      out << "FAILURE: fm_err " << fm_err << " > tol " << tol << endl;
      out << "fm expected: " << fm->displayString() << endl;
      out << "fm actual:   " << exact_fm[0]->displayString() << endl;
    }
    
    double fe_err = (fe - exact_fe)->l2norm(mesh, cubatureEnrichment);
    if (fe_err > tol)
    {
      success = false;
      out << "FAILURE: fe_err " << fe_err << " > tol " << tol << endl;
      out << "fe expected: " << fe->displayString() << endl;
      out << "fe actual:   " << exact_fe->displayString() << endl;
    }
  }
  
  void testResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 1;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = 1e2; // Reynolds number
    
    bool useConformingTraces = true;
    auto form = CompressibleNavierStokesFormulationRefactor::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                               meshTopo, polyOrder, delta_k);
    
    BFPtr bf = form.bf();
    RHSPtr rhs = form.rhs();
    
    auto soln = form.solution();
    auto solnIncrement = form.solutionIncrement();
    
    auto exactMap = form.exactSolutionMap(u, rho, T);
    auto f_c = form.exactSolution_fc(u, rho, T);
    auto f_m = form.exactSolution_fm(u, rho, T);
    auto f_e = form.exactSolution_fe(u, rho, T);
    
    form.setForcing(f_c, f_m, f_e);
    
    auto vf = bf->varFactory();
    // split the exact solution into traces and fields
    // we want to project the traces onto the increment, and the fields onto the background flow (soln)
    map<int, FunctionPtr> traceMap, fieldMap;
    for (auto entry : exactMap)
    {
      int trialID = entry.first;
      FunctionPtr f = entry.second;
      if (vf->trial(trialID)->isDefinedOnVolume()) // field
      {
        fieldMap[trialID] = f;
      }
      else
      {
        traceMap[trialID] = f;
      }
    }
    int solnOrdinal = 0; // no goal-oriented stuff here...
    soln->projectOntoMesh(fieldMap, solnOrdinal);
//    solnIncrement->projectOntoMesh(traceMap, solnOrdinal);
    
    auto residual = bf->testFunctional(traceMap) - rhs->linearTerm();
    
    auto testIP = solnIncrement->ip(); // We'll use this inner product to take the norm of the residual components
    
    auto summands = residual->summands();
    auto testVars = vf->testVars();
    for (auto testEntry : testVars)
    {
      int testID = testEntry.first;
      VarPtr testVar = testEntry.second;
      // filter the parts of residual that involve testVar
      LinearTermPtr testResidual = Teuchos::rcp( new LinearTerm );
      for (auto summandEntry : summands)
      {
        FunctionPtr f = summandEntry.first;
        VarPtr v = summandEntry.second;
        if (v->ID() == testID)
        {
          testResidual = testResidual + f * v;
        }
      }
      double residualNorm = testResidual->computeNorm(testIP, soln->mesh(), cubatureEnrichment);
      if (residualNorm > tol)
      {
        success = false;
        out << "FAILURE: residual in " << testVar->name() << " component: " << residualNorm << " exceeds tolerance " << tol << ".\n";
        out << "Residual string: " << testResidual->displayString() << "\n";
        out << "Exact solution under test:\n";
        for (auto traceEntry : traceMap)
        {
          int trialID = traceEntry.first;
          std::string varName = vf->trial(trialID)->name();
          out << "  " << varName << ": " << traceEntry.second->displayString() << std::endl;
        }
        for (auto trialEntry : fieldMap)
        {
          int trialID = trialEntry.first;
          std::string varName = vf->trial(trialID)->name();
          out << "  " << varName << ": " << trialEntry.second->displayString() << std::endl;
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_AllZero)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero(); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_LinearTemp)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::xn(1);
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero(); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(1.0); // expected forcing for momentum equation
    FunctionPtr f_e = Function::constant(0.5); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_LinearVelocity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    double fe_const = -4./3. * 1. / TEST_RE;
    FunctionPtr f_e = Function::constant(fe_const); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_LinearTempUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    FunctionPtr f_c = Function::zero();            // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(TEST_R);  // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero();            // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);;
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = 2.0 * x;                 // expected forcing for momentum equation
    FunctionPtr f_e = 1.5 * x * x + -4./3. * 1. / TEST_RE; // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Forcing_1D_Steady_QuadraticTemp)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = x * x;
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::constant(-TEST_CP * 2.0 / (TEST_PR * TEST_RE)); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_AllZero)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_AllOne)
  {
    double tol = 1e-13;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_UnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_UnitTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::constant(1.0);
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_UnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_LinearDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_LinearTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::xn(1);
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulationRefactor, Residual_1D_Steady_LinearVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, Consistency_Steady_2D )
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 1.0e2;
//    int fieldPolyOrder = 3, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    //    FunctionPtr u1 = x;
//    //    FunctionPtr u2 = -y; // divergence 0
//    //    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = x * x * y;
//    FunctionPtr u2 = -x * y * y;
//    FunctionPtr p = y * y * y;
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
////    cout << "forcingFunction: " << forcingFunction->displayString() << endl;
//    form.setForcingFunction(forcingFunction);
//
//    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
//    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
//
//    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
//    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
//
//    map<int, FunctionPtr> exactMap;
//    // fields:
//    exactMap[form.u(1)->ID()] = u1;
//    exactMap[form.u(2)->ID()] = u2;
//    exactMap[form.p()->ID() ] =  p;
//    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
//    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
//    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
//    exactMap[form.sigma(2,2)->ID()] = sigma2->y();
//
//    // fluxes:
//    // use the exact field variable solution together with the termTraced to determine the flux traced
//    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
//    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
//    exactMap[form.tn_hat(1)->ID()] = t1_n;
//    exactMap[form.tn_hat(2)->ID()] = t2_n;
//
//    // traces:
//    exactMap[form.u_hat(1)->ID()] = u1;
//    exactMap[form.u_hat(2)->ID()] = u2;
//
//    map<int, FunctionPtr> zeroMap;
//    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
//    {
//      zeroMap[exactMapIt->first] = Function::zero(exactMapIt->second->rank());
//    }
//
//    const int solutionOrdinal = 0;
//    form.solution()->projectOntoMesh(exactMap, solutionOrdinal);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal);
//
//    RHSPtr rhs = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
//    form.solutionIncrement()->setRHS(rhs);
//
////    cout << "rhs: " << rhs->linearTerm()->displayString() << endl;
//
//    double energyError = form.solutionIncrement()->energyErrorTotal();
//
//    double tol = 1e-13;
//    TEST_COMPARE(energyError, <, tol);
//  }
//
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, Consistency_SteadyConservation_2D )
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 1.0e2;
//    int fieldPolyOrder = 3, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    //    FunctionPtr u1 = x;
//    //    FunctionPtr u2 = -y; // divergence 0
//    //    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = x * x * y;
//    FunctionPtr u2 = -x * y * y;
//    FunctionPtr p = y * y * y;
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyConservationFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
////    cout << "forcingFunction: " << forcingFunction->displayString() << endl;
//    form.setForcingFunction(forcingFunction);
//
//    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
//    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
//
//    // LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
//    // LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
//
//    map<int, FunctionPtr> exactMap;
//    // fields:
//    exactMap[form.u(1)->ID()] = u1;
//    exactMap[form.u(2)->ID()] = u2;
//    exactMap[form.p()->ID() ] =  p;
//    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
//    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
//    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
//    exactMap[form.sigma(2,2)->ID()] = sigma2->y();
//
//    // fluxes:
//    // use the exact field variable solution together with the termTraced to determine the flux traced
//    FunctionPtr n_x = Function::normal();
//    FunctionPtr n_x_parity = n_x * TFunction<double>::sideParity();
//    FunctionPtr t1_n = u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y() - sigma1*n_x_parity + p*n_x_parity->x();
//    FunctionPtr t2_n = u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y() - sigma2*n_x_parity + p*n_x_parity->y();
//    // FunctionPtr t1_n = t1_n_lt->evaluate(exactMap) + u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y();
//    // FunctionPtr t2_n = t2_n_lt->evaluate(exactMap) + u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y();
//    exactMap[form.tn_hat(1)->ID()] = t1_n;
//    exactMap[form.tn_hat(2)->ID()] = t2_n;
//
//    // traces:
//    exactMap[form.u_hat(1)->ID()] = u1;
//    exactMap[form.u_hat(2)->ID()] = u2;
//
//    map<int, FunctionPtr> zeroMap;
//    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
//    {
//      zeroMap[exactMapIt->first] = Function::zero(exactMapIt->second->rank());
//    }
//
//    const int solutionOrdinal = 0;
//    form.solution()->projectOntoMesh(exactMap, solutionOrdinal);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal);
//
//    RHSPtr rhs = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
//    form.solutionIncrement()->setRHS(rhs);
//
////    cout << "rhs: " << rhs->linearTerm()->displayString() << endl;
//
//    double energyError = form.solutionIncrement()->energyErrorTotal();
//
//    double tol = 1e-13;
//    TEST_COMPARE(energyError, <, tol);
//  }
//
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, ExactSolution_Steady_2D_Slow )
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 1e2;
//    int fieldPolyOrder = 3, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//
//    FunctionPtr u1 = x * x * y;
//    FunctionPtr u2 = -x * y * y;
//    FunctionPtr p = y;
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
//    form.setForcingFunction(forcingFunction);
//    RHSPtr rhsForSolve = form.solutionIncrement()->rhs();
//
////    cout << "bf for Navier-Stokes:\n";
////    form.bf()->printTrialTestInteractions();
//
////    cout << "rhs for Navier-Stokes solve:\n" << rhsForSolve->linearTerm()->displayString();
//
//    form.addInflowCondition(SpatialFilter::allSpace(), Function::vectorize(u1, u2));
//    form.addZeroMeanPressureCondition();
//
//    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
//    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
//
//    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
//    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
//
//    map<int, FunctionPtr> exactMap;
//    // fields:
//    exactMap[form.u(1)->ID()] = u1;
//    exactMap[form.u(2)->ID()] = u2;
//    exactMap[form.p()->ID() ] =  p;
//    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
//    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
//    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
//    exactMap[form.sigma(2,2)->ID()] = sigma2->y();
//
//    // fluxes:
//    // use the exact field variable solution together with the termTraced to determine the flux traced
//    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
//    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
//    exactMap[form.tn_hat(1)->ID()] = t1_n;
//    exactMap[form.tn_hat(2)->ID()] = t2_n;
//
//    // traces:
//    exactMap[form.u_hat(1)->ID()] = u1;
//    exactMap[form.u_hat(2)->ID()] = u2;
//
//    map<int, FunctionPtr> zeroMap;
//    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
//    {
//      VarPtr trialVar = form.bf()->varFactory()->trial(exactMapIt->first);
//      FunctionPtr zero = Function::zero();
//      for (int i=0; i<trialVar->rank(); i++)
//      {
//        if (spaceDim == 2)
//          zero = Function::vectorize(zero, zero);
//        else if (spaceDim == 3)
//          zero = Function::vectorize(zero, zero, zero);
//      }
//      zeroMap[exactMapIt->first] = zero;
//    }
//
//    RHSPtr rhsWithBoundaryTerms = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
//
//    double tol = 1e-10;
//
//    // sanity/consistency check: is the energy error for a zero solutionIncrement zero?
//    const int solutionOrdinal = 0;
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal);
//    form.solution()->projectOntoMesh(exactMap, solutionOrdinal);
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    double energyError = form.solutionIncrement()->energyErrorTotal();
//
//    TEST_COMPARE(energyError, <, tol);
//
//    // change RHS back for solve below:
//    form.solutionIncrement()->setRHS(rhsForSolve);
//
//    // first real test: with exact background flow, if we solve, do we maintain zero energy error?
//    form.solveAndAccumulate();
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal); // zero out since we've accumulated
//    energyError = form.solutionIncrement()->energyErrorTotal();
//    TEST_COMPARE(energyError, <, tol);
//
//    // change RHS back for solve below:
//    form.solutionIncrement()->setRHS(rhsForSolve);
//
//    // next test: try starting from a zero initial guess
//    form.solution()->projectOntoMesh(zeroMap, solutionOrdinal);
//
//    SolutionPtr solnIncrement = form.solutionIncrement();
//
//    FunctionPtr u1_incr = Function::solution(form.u(1), solnIncrement);
//    FunctionPtr u2_incr = Function::solution(form.u(2), solnIncrement);
//    FunctionPtr p_incr = Function::solution(form.p(), solnIncrement);
//
//    FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr;
//
//    double l2_norm_incr = 0.0;
//    double nonlinearTol = 1e-12;
//    int maxIters = 10;
//    do
//    {
//      form.solveAndAccumulate();
//      l2_norm_incr = sqrt(l2_incr->integrate(solnIncrement->mesh()));
//      out << "iteration " << form.nonlinearIterationCount() << ", L^2 norm of increment: " << l2_norm_incr << endl;
//    }
//    while ((l2_norm_incr > nonlinearTol) && (form.nonlinearIterationCount() < maxIters));
//
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal); // zero out since we've accumulated
//    energyError = form.solutionIncrement()->energyErrorTotal();
//    TEST_COMPARE(energyError, <, tol);
//
//    //    if (energyError >= tol) {
//    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_background_flow",form.solution());
//    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_soln_increment",form.solutionIncrement());
//    //    }
//  }
//
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, ExactSolution_SteadyConservation_2D_Slow )
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 1.0;
//    int fieldPolyOrder = 3, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//
//    FunctionPtr u1 = x * x * y;
//    FunctionPtr u2 = -x * y * y;
//    FunctionPtr p = y;
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyConservationFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
//    form.setForcingFunction(forcingFunction);
//    RHSPtr rhsForSolve = form.solutionIncrement()->rhs();
//
////    cout << "bf for Navier-Stokes:\n";
////    form.bf()->printTrialTestInteractions();
//
////    cout << "rhs for Navier-Stokes solve:\n" << rhsForSolve->linearTerm()->displayString();
//
//    form.addInflowCondition(SpatialFilter::allSpace(), Function::vectorize(u1, u2));
//    form.addZeroMeanPressureCondition();
//
//    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
//    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
//
//    // LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
//    // LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
//
//    map<int, FunctionPtr> exactMap;
//    // fields:
//    exactMap[form.u(1)->ID()] = u1;
//    exactMap[form.u(2)->ID()] = u2;
//    exactMap[form.p()->ID() ] =  p;
//    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
//    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
//    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
//    exactMap[form.sigma(2,2)->ID()] = sigma2->y();
//
//    // fluxes:
//    // use the exact field variable solution together with the termTraced to determine the flux traced
//    FunctionPtr n_x = Function::normal();
//    FunctionPtr n_x_parity = n_x * TFunction<double>::sideParity();
//    FunctionPtr t1_n = u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y() - sigma1*n_x_parity + p*n_x_parity->x();
//    FunctionPtr t2_n = u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y() - sigma2*n_x_parity + p*n_x_parity->y();
//    // FunctionPtr t1_n = t1_n_lt->evaluate(exactMap) + u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y();
//    // FunctionPtr t2_n = t2_n_lt->evaluate(exactMap) + u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y();
//    exactMap[form.tn_hat(1)->ID()] = t1_n;
//    exactMap[form.tn_hat(2)->ID()] = t2_n;
//
//    // traces:
//    exactMap[form.u_hat(1)->ID()] = u1;
//    exactMap[form.u_hat(2)->ID()] = u2;
//
//    map<int, FunctionPtr> zeroMap;
//    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
//    {
//      VarPtr trialVar = form.bf()->varFactory()->trial(exactMapIt->first);
//      FunctionPtr zero = Function::zero();
//      for (int i=0; i<trialVar->rank(); i++)
//      {
//        if (spaceDim == 2)
//          zero = Function::vectorize(zero, zero);
//        else if (spaceDim == 3)
//          zero = Function::vectorize(zero, zero, zero);
//      }
//      zeroMap[exactMapIt->first] = zero;
//    }
//
//    RHSPtr rhsWithBoundaryTerms = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
//
//    double tol = 1e-12;
//
//    // sanity/consistency check: is the energy error for a zero solutionIncrement zero?
//    const int solutionOrdinal = 0;
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal);
//    form.solution()->projectOntoMesh(exactMap, solutionOrdinal);
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    double energyError = form.solutionIncrement()->energyErrorTotal();
//
//    TEST_COMPARE(energyError, <, tol);
//
//    // change RHS back for solve below:
//    form.solutionIncrement()->setRHS(rhsForSolve);
//
//    // first real test: with exact background flow, if we solve, do we maintain zero energy error?
//    form.solveAndAccumulate();
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal); // zero out since we've accumulated
//    energyError = form.solutionIncrement()->energyErrorTotal();
//    TEST_COMPARE(energyError, <, tol);
//
//    // change RHS back for solve below:
//    form.solutionIncrement()->setRHS(rhsForSolve);
//
//    // next test: try starting from a zero initial guess
//    form.solution()->projectOntoMesh(zeroMap, solutionOrdinal);
//
//    SolutionPtr solnIncrement = form.solutionIncrement();
//
//    FunctionPtr u1_incr = Function::solution(form.u(1), solnIncrement);
//    FunctionPtr u2_incr = Function::solution(form.u(2), solnIncrement);
//    FunctionPtr p_incr = Function::solution(form.p(), solnIncrement);
//
//    FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr;
//
//    double l2_norm_incr = 0.0;
//    double nonlinearTol = 1e-12;
//    int maxIters = 20;
//    do
//    {
//      form.solveAndAccumulate();
//      l2_norm_incr = sqrt(l2_incr->integrate(solnIncrement->mesh()));
//      out << "iteration " << form.nonlinearIterationCount() << ", L^2 norm of increment: " << l2_norm_incr << endl;
//    }
//    while ((l2_norm_incr > nonlinearTol) && (form.nonlinearIterationCount() < maxIters));
//
//    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
//    form.solutionIncrement()->projectOntoMesh(zeroMap, solutionOrdinal); // zero out since we've accumulated
//    energyError = form.solutionIncrement()->energyErrorTotal();
//    TEST_COMPARE(energyError, <, tol);
//
//    //    if (energyError >= tol) {
//    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_background_flow",form.solution());
//    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_soln_increment",form.solutionIncrement());
//    //    }
//  }
//
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, ForcingFunction_Steady_2D)
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 1.0e2;
//    int fieldPolyOrder = 3, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//
//    FunctionPtr u1 = x * x * y;
//    FunctionPtr u2 = -x * y * y;
//    FunctionPtr p = y * y * y;
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
//    
//    FunctionPtr expectedForcingFunction_x = p->dx() - (1.0 / Re) * (u1->dx()->dx() + u1->dy()->dy()) + u1 * u1->dx() + u2 * u1->dy();
//    FunctionPtr expectedForcingFunction_y = p->dy() - (1.0 / Re) * (u2->dx()->dx() + u2->dy()->dy()) + u1 * u2->dx() + u2 * u2->dy();
//    
//    double err_x = (expectedForcingFunction_x - forcingFunction->x())->l2norm(form.solution()->mesh());
//    double err_y = (expectedForcingFunction_y - forcingFunction->y())->l2norm(form.solution()->mesh());
//
//    double tol = 1e-12;
//    TEST_COMPARE(err_x, <, tol);
//    TEST_COMPARE(err_y, <, tol);
////    cout << forcingFunction->displayString();
//  }
//
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, SpaceTimeConservation_FormMeshAgreesWithManualMesh )
//  {
//    double pi = atan(1)*4;
//    vector<double> x0 = {0.0, 0.0};;
//    vector<double> dims = {2*pi, 2*pi};
//    vector<int> numElements = {2,2};
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
//    double t0 = 0;
//    double t1 = pi;
//    int temporalDivisions = 1;
//    meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
//    // some refinements in an effort to replicate an issue, which may be revealed in a difference between
//    // the dofs assigned to a MeshTopology and a MeshTopologyView with the same elements
//    // 1. Uniform refinement
//    IndexType nextElement = meshTopo->cellCount();
//    vector<IndexType> cellsToRefine = meshTopo->getActiveCellIndicesGlobal();
//    CellTopoPtr cellTopo = meshTopo->getCell(0)->topology();
//    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
//    for (IndexType cellIndex : cellsToRefine)
//    {
//      meshTopo->refineCell(cellIndex, refPattern, nextElement);
//      nextElement += refPattern->numChildren();
//    }
//    // 2. Selective refinement
//    cellsToRefine = {4,15,21,30};
//    for (IndexType cellIndex : cellsToRefine)
//    {
//      meshTopo->refineCell(cellIndex, refPattern, nextElement);
//      nextElement += refPattern->numChildren();
//    }
//    
//    int fieldPolyOrder = 1, delta_k = 1;
//    int spaceDim = x0.size();
//    double Re = 1.0;
//    bool useConformingTraces = true;
//
////    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::spaceTimeConservationFormulation(spaceDim, Re, useConformingTraces,
////                                                                                                   meshTopo, fieldPolyOrder, fieldPolyOrder, delta_k);
//    
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::spaceTimeFormulation(spaceDim, Re, useConformingTraces,
//                                                                                       meshTopo, fieldPolyOrder, fieldPolyOrder, delta_k);
//    
//    MeshPtr formMesh = form.solutionIncrement()->mesh(); //Teuchos::rcp( new Mesh(meshTopo, form.bf(), fieldPolyOrder+1, delta_k) ) ;
//    vector<int> H1Order = {fieldPolyOrder + 1, fieldPolyOrder + 1};
//    MeshPtr manualMesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) ) ;
//    
//    GlobalIndexType numGlobalDofsFormMesh = formMesh->numGlobalDofs();
//    GlobalIndexType numGlobalDofsManualMesh = manualMesh->numGlobalDofs();
//    
////    cout << "numGlobalDofsFormMesh: " << numGlobalDofsFormMesh << endl;
//    
//    TEST_EQUALITY(numGlobalDofsManualMesh, numGlobalDofsFormMesh);
//  }
//  
//  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, StokesConsistency_Steady_2D )
//  {
//    int spaceDim = 2;
//    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
//    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
//    vector<double> x0(spaceDim,-1.0);
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
//    double Re = 10.0;
//    int fieldPolyOrder = 2, delta_k = 1;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    //    FunctionPtr u1 = x;
//    //    FunctionPtr u2 = -y; // divergence 0
//    //    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = x;
//    FunctionPtr u2 = -y;
//    FunctionPtr p = y;
//
//    FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
//    FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
//    FunctionPtr forcingFunction = Function::vectorize(forcingFunction_x, forcingFunction_y);
//
//    bool useConformingTraces = true;
//    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
//    form.setForcingFunction(forcingFunction);
//
//    BFPtr stokesBF = form.stokesBF();
//
//    MeshPtr stokesMesh = Teuchos::rcp( new Mesh(meshTopo,stokesBF,fieldPolyOrder+1, delta_k) );
//
//    SolutionPtr stokesSolution = Solution::solution(stokesMesh);
//    stokesSolution->setIP(stokesBF->graphNorm());
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm(forcingFunction_x * form.v(1));
//    rhs->addTerm(forcingFunction_y * form.v(2));
//
//    stokesSolution->setRHS(rhs);
//
//    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
//    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
//
//    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
//    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
//
//    map<int, FunctionPtr> exactMap;
//    // fields:
//    exactMap[form.u(1)->ID()] = u1;
//    exactMap[form.u(2)->ID()] = u2;
//    exactMap[form.p()->ID() ] =  p;
//    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
//    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
//    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
//    exactMap[form.sigma(2,2)->ID()] = sigma2->y();
//
//    // fluxes:
//    // use the exact field variable solution together with the termTraced to determine the flux traced
//    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
//    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
//    exactMap[form.tn_hat(1)->ID()] = t1_n;
//    exactMap[form.tn_hat(2)->ID()] = t2_n;
//
//    // traces:
//    exactMap[form.u_hat(1)->ID()] = u1;
//    exactMap[form.u_hat(2)->ID()] = u2;
//
//    const int solutionOrdinal = 0;
//    stokesSolution->projectOntoMesh(exactMap, solutionOrdinal);
//
//    double energyError = stokesSolution->energyErrorTotal();
//
//    double tol = 1e-14;
//    TEST_COMPARE(energyError, <, tol);
//  }
} // namespace
