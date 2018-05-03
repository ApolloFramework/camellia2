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

#include "MeshFactory.h"
#include "IdealMHDFormulation.hpp"
#include "HDF5Exporter.h"
#include "RHS.h"
#include "SimpleFunction.h"
#include "Solution.h"

using namespace Camellia;
using namespace Intrepid;

#include <array>

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  static const double TEST_RE = 1e2;
  static const double TEST_PR = 0.713;
  static const double TEST_CV = 1.000;
  static const double TEST_GAMMA = 1.4;
  static const double TEST_CP = TEST_GAMMA * TEST_CV;
  static const double TEST_R = TEST_CP - TEST_CV;
  
  static const double DEFAULT_RESIDUAL_TOLERANCE = 1e-12; // now that we do a solve, need to be a bit more relaxed
  static const double DEFAULT_NL_SOLVE_TOLERANCE =  1e-6; // for nonlinear stepping
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_1D)
  {
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
