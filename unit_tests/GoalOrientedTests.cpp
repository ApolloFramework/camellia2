//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TestTemplate
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

// empty test file.  Copy (naming "MyClassTests.cpp", typically) and then add your tests below.

#include "Teuchos_UnitTestHarness.hpp"

#include "EpetraExt_MultiVectorOut.h"

#include "Camellia.h"

using namespace Camellia;

namespace
{
  MeshPtr poissonUniformMesh(const vector<int> & elementWidths, int H1Order, bool useConformingTraces)
  {
    int spaceDim = elementWidths.size();
    int testSpaceEnrichment = spaceDim; //
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(spaceDim,span);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
    return mesh;
  }
  
  SolutionPtr simplePoissonSolution(const vector<int> &elementWidths, int H1Order, bool useConformingTraces)
  {
    // unit forcing, homogeneous BCs
    auto mesh = poissonUniformMesh(elementWidths, H1Order, useConformingTraces);
    const int spaceDim = elementWidths.size();
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    auto rhs = poissonForm.rhs(Function::constant(1.0));
    auto u_hat = poissonForm.u_hat();
    auto bc = BC::bc();
    bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
    auto graphNorm = poissonForm.bf()->graphNorm();
    auto soln = Solution::solution(poissonForm.bf(),mesh,bc,rhs,graphNorm);
    
    return soln;
  }
  
  SolutionPtr simplePoissonSolutionWithSecondaryRHS(const vector<int> &elementWidths, int H1Order, bool useConformingTraces)
  {
    auto soln = simplePoissonSolution(elementWidths, H1Order, useConformingTraces);
    const int spaceDim = elementWidths.size();
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    
    LinearTermPtr g = 1*u;
    soln->setGoalOrientedRHS(g);
    
    return soln;
  }
  
  void testPoissonSolveMatches(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, double tol, bool &success, Teuchos::FancyOStream &out)
  {
    // simple test that Poisson solve with secondary ("goal-oriented") RHS
    // still (a) has the same primary solution
    // and   (b) gets the same adaptive refinements
    
    const int spaceDim = elementWidths.size();
    
    auto soln_OneRHS = simplePoissonSolution                (elementWidths, H1Order, useConformingTraces);
    auto soln_TwoRHS = simplePoissonSolutionWithSecondaryRHS(elementWidths, H1Order, useConformingTraces);
    
    //    {
    //      // DEBUGGING: write out the matrices and RHSes to file
    //      soln_OneRHS->setWriteMatrixToMatrixMarketFile(true, "/tmp/A_one.dat");
    //      soln_TwoRHS->setWriteMatrixToMatrixMarketFile(true, "/tmp/A_two.dat");
    //      soln_OneRHS->setWriteRHSToMatrixMarketFile(true, "/tmp/b_one.dat");
    //      soln_TwoRHS->setWriteRHSToMatrixMarketFile(true, "/tmp/b_two.dat");
    //    }
    
    soln_OneRHS->solve();
    soln_TwoRHS->solve();
    
    //    {
    //      // DEBUGGING:
    //      auto lhs_OneRHS = soln_OneRHS->getLHSVector();
    //      auto lhs_TwoRHS = soln_TwoRHS->getLHSVector();
    //      bool includeHeaders = true;
    //
    //      EpetraExt::MultiVectorToMatrixMarketFile("/tmp/x_one.dat",*lhs_OneRHS,0,0,includeHeaders);
    //      EpetraExt::MultiVectorToMatrixMarketFile("/tmp/x_two.dat",*lhs_TwoRHS,0,0,includeHeaders);
    //    }
    
    // we want to check that the (primary) solutions match each other
    // we'll do some higher-level tests below, but to start with, we'll check that the solution coefficients match
    auto lhs_OneRHS = soln_OneRHS->getLHSVector();
    auto lhs_TwoRHS = soln_TwoRHS->getLHSVector(); // the first column of this should match the first column of lhs_OneRHS
    
    int localLength = lhs_OneRHS->MyLength();
    TEST_EQUALITY(localLength, lhs_TwoRHS->MyLength()); // the value distribution across MPI should also match
    
    // we take advantage of the fact that the value distribution should match here
    for (int i=0; i<localLength; i++)
    {
      auto oneRHSValue = (*lhs_OneRHS)[0][i];
      auto twoRHSValue = (*lhs_TwoRHS)[0][i];
      // also use tol as a ceiling for rounding to zero:
      if ((abs(oneRHSValue) < tol) && ((abs(twoRHSValue) < tol)))
      {
        continue;
      }
      TEST_FLOATING_EQUALITY(oneRHSValue, twoRHSValue, tol);
    }
    
    // Solution *also* stores a cell-local representation of solution coefficients, and this should also match
    // for the first (0) solution ordinals.
    auto & myCellIDs = soln_OneRHS->mesh()->cellIDsInPartition();
    for (auto cellID : myCellIDs)
    {
      bool warnAboutOffRank = true; // should all be on-rank
      const int solutionOrdinal = 0;
      auto & coeffsOne = soln_OneRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,solutionOrdinal);
      auto & coeffsTwo = soln_TwoRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,solutionOrdinal);
      if (coeffsOne.size() != coeffsTwo.size())
      {
        out << "FAILURE: Sizes differ: coeffsOne is of length " << coeffsOne.size();
        out << ", while coeffsTwo is of length " << coeffsTwo.size() << endl;
        success = false;
      }
      else
      {
        int dofCount = coeffsOne.size();
        bool savedSuccess = success; // allows us to detect local failure
        success = true;
        for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
        {
          auto dofOne = coeffsOne[dofOrdinal];
          auto dofTwo = coeffsTwo[dofOrdinal];
          if ((abs(dofOne) < tol) && (abs(dofTwo) < tol))
          {
            // both zero, essentially
            continue;
          }
          else
          {
            TEST_FLOATING_EQUALITY(dofOne, dofTwo, tol);
          }
        }
        if (!success) // local failure
        {
          out << "Dofs do not match on cell ID " << cellID << endl;
          out << "coeffsOne:\n" << coeffsOne;
          out << "coeffsTwo:\n" << coeffsTwo;
        }
        // copy back saved value of success
        success = savedSuccess;
      }
    }
    
    // higher-level check below -- check that the solutions globally match
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    
    auto u_OneRHS = Function::solution(u, soln_OneRHS);
    auto u_TwoRHS = Function::solution(u, soln_TwoRHS);
    auto u_diff   = u_OneRHS - u_TwoRHS;
    
    auto u_err_L2 = u_diff->l2norm(soln_OneRHS->mesh());
    TEST_COMPARE(u_err_L2, <, tol);
    
    /*
     Right now, this test fails.  It does appear that the linear algebra is done correctly; it
     seems likely that SimpleSolutionFunction or something else downstream of the Solution coefficients
     is doing something wrong (what, exactly, I'm not sure).
     */
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_1D )
  {
    const int spaceDim = 1;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 4;
    bool useConformingTraces = true;
    double tol = 1e-16;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_2D )
  {
    const int spaceDim = 2;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 3;
    bool useConformingTraces = true;
    double tol = 1e-16;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_3D )
  {
    const int spaceDim = 3;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    double tol = 1e-16;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, tol, success, out);
  }
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace
