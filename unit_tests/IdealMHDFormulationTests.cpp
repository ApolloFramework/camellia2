//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CompressibleNavierStokesConservationFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/9/18.
//
//

#include "Functions.hpp"
#include "MeshFactory.h"
#include "IdealMHDFormulation.hpp"
#include "HDF5Exporter.h"
#include "RHS.h"
#include "RieszRep.h"
#include "SimpleFunction.h"
#include "Solution.h"

using namespace Camellia;
using namespace Intrepid;

#include <array>

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  void testTermsMatch(LinearTermPtr ltExpected, LinearTermPtr ltActual, MeshPtr mesh, IPPtr ip, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    LinearTermPtr diff = ltExpected - ltActual;
    
    BFPtr bf = mesh->bilinearForm();
    
    RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, diff) );
    
    rieszRep->computeRieszRep();
    
    double err = rieszRep->getNorm();
    
    TEST_COMPARE(err, <, tol);
  }
  
  static const double TEST_CV = 1.000;
  static const double TEST_GAMMA = 2.0;
  static const double TEST_CP = TEST_GAMMA * TEST_CV;
  static const double TEST_R = TEST_CP - TEST_CV;
  
  static const double DEFAULT_RESIDUAL_TOLERANCE = 1e-12; // now that we do a solve, need to be a bit more relaxed
  static const double DEFAULT_NL_SOLVE_TOLERANCE =  1e-6; // for nonlinear stepping
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_1D)
  {
    double tol = 1e-14;
    int spaceDim = 1;
    double x_a = 0.0, x_b = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 7); // 1D: vc, ve, plus 3 vm's, 2 vBs
    
    int numTrialVars = vf->trialVars().size(); // 1D: rho, E, 3 m's, 2 B's, plus 1 flux per equation: 14 variables
    TEST_EQUALITY(numTrialVars, 14);
    
    auto bf = form->solution()->bf();
    // try momentum = 0, E = 1, rho = 1, B = 0 -- project this onto previous solution, and current guess
    // evaluate bf at this solution
    // we expect to have just five test terms survive:
    // three corresponding to the pressure in the momentum equation
    // two corresponding to the time derivatives on the rho, E solution increments
    VarPtr vm1 = form->vm(1);
    VarPtr vm2 = form->vm(2);
    VarPtr vm3 = form->vm(3);
    VarPtr vc  = form->vc();
    VarPtr ve  = form->ve();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E = Function::constant(1.0);
    FunctionPtr m = Function::vectorize(Function::constant(0.0), Function::constant(0.0), Function::constant(0.0));
    FunctionPtr p = (TEST_GAMMA - 1.0) * (E - 0.5 * dot(3,m,m) / rho);
    auto dt = form->getTimeStep();
    auto expectedLT = p * vm1 + p * vm2 + p * vm3 + rho / dt * vc + E / dt * ve;
    
    map<int, FunctionPtr> valueMap;
    valueMap[form->rho()->ID()] = rho;
    valueMap[form->E()->ID()] = E;
    valueMap[form->m(1)->ID()] = m->spatialComponent(1);
    valueMap[form->m(2)->ID()] = m->spatialComponent(2);
    valueMap[form->m(3)->ID()] = m->spatialComponent(3);
    
    int solutionOrdinal = 0;
    form->solutionPreviousTimeStep()->projectOntoMesh(valueMap, solutionOrdinal);
    form->solution()->projectOntoMesh(valueMap, solutionOrdinal);
    auto lt = bf->testFunctional(valueMap);
    out << "lt: " << lt->displayString() << endl;
    
    auto mesh = form->solution()->mesh();
    auto ip = form->solutionIncrement()->ip();
    
    ip->printInteractions();
//    cout << "IP: " << ip->displayString() << endl;
    testTermsMatch(expectedLT, lt, mesh, ip, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_2D)
  {
    int spaceDim = 2;
    double width = 1.0, height = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({width, height}, {meshWidth, meshWidth});
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 9); // 2D: vc, ve, plus 3 vm's, 3 vBs, plus vGauss
    
    int numTrialVars = vf->trialVars().size(); // 2D: rho, E, 3 m's, 3 B's, plus 1 flux per equation: 17 variables
    TEST_EQUALITY(numTrialVars, 17);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_3D)
  {
    int spaceDim = 3;
    double width = 1.0, height = 1.0, depth = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({width, height, depth}, {meshWidth, meshWidth, meshWidth});
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 9); // 2D: vc, ve, plus 3 vm's, 3 vBs, plus vGauss
    
    int numTrialVars = vf->trialVars().size(); // 2D: rho, E, 3 m's, 3 B's, plus 1 flux per equation: 17 variables
    TEST_EQUALITY(numTrialVars, 17);
  }
} // namespace
