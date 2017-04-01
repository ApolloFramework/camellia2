//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  DLSTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/30/17.
//
//

// Int_UnitTests.cpp
#include "Teuchos_UnitTestHarness.hpp"

#include "DLS.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TypeDefs.h"

#include "Epetra_LocalMap.h"

using namespace Camellia;
using namespace Teuchos;
using namespace std;

namespace
{
  typedef Teuchos::RCP<DLS<double>> DLSPtr;
  void setupSolve(SolutionPtr &standard, DLSPtr &dls)
  {
    int spaceDim = 2;
    bool useConformingTraces = true;
    int meshWidth = 2;
    double domainWidth = 1.0;
    int testSpaceEnrichment = 2;
    int H1Order = 2;
    
    vector<double> dimensions(spaceDim, domainWidth);
    vector<int> elementWidths(spaceDim,meshWidth);
    
    PoissonFormulation form(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dimensions, elementWidths,
                                                H1Order, testSpaceEnrichment);

    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(1.0 * form.v());
    
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), Function::zero());
    
    standard = Solution::solution(mesh,bc,rhs,form.bf()->graphNorm());
    standard->initializeLHSVector();
    standard->initializeStiffnessAndLoad();
    standard->populateStiffnessAndLoad();
    
    SolutionPtr dlsSolution = Solution::solution(mesh,bc,rhs,form.bf()->graphNorm());
    dls = Teuchos::rcp( new DLS<double>(dlsSolution) );
    dls->assemble();
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonDiagonalsAgree )
  {
    DLSPtr dls;
    SolutionPtr standardSolution;
    setupSolve(standardSolution, dls);
    
    TVectorPtr<double> dlsInverseSqrtDiag = dls->normalDiagInverseSqrt();
    
    Epetra_CommPtr comm = standardSolution->mesh()->Comm();
    
    GlobalIndexType standardSolutionDofCount = standardSolution->getDofInterpreter()->globalDofCount();
    GlobalIndexType indexBase = 0;
    // we'll replicate standard solution diagonal on every rank:
    Epetra_LocalMap localMap(standardSolutionDofCount, indexBase, *comm);
    Epetra_Vector normalDiag(localMap);
    standardSolution->getStiffnessMatrix()->ExtractDiagonalCopy(normalDiag);
    
    // compare:
    double tol = 1e-14;
    using namespace Tpetra;
    typedef Map<IndexType,GlobalIndexType> Map;
    typedef RCP<const Map> constMapRCP;
    constMapRCP dlsMap = dlsInverseSqrtDiag->getMap();
    size_t myDLSEntries = dlsMap->getNodeNumElements();
    
    auto dlsInverseSqrtDiagValues_2d = dlsInverseSqrtDiag->template getLocalView<Kokkos::HostSpace> ();
    // getLocalView returns a 2-D View by default.  We want a 1-D View, so we take a subview.
    auto dlsInverseSqrtDiagValues_1d = Kokkos::subview (dlsInverseSqrtDiagValues_2d, Kokkos::ALL (), 0);
    
    double maxErr = 0.0;
    for (size_t localID = 0; localID < myDLSEntries; localID++)
    {
      GlobalIndexType GID = dlsMap->getGlobalElement(localID);
      IndexType standardLocalID = localMap.LID(GID); // should be the same as GID
      double normalDiagValue = normalDiag[standardLocalID];
      double inverseSqrtDiag = dlsInverseSqrtDiagValues_1d(localID);

//      cout << "inverseSqrtDiag for GID " << GID << ": " << inverseSqrtDiag << endl;
      
      double expected = normalDiagValue;
      double actual = 1.0 / (inverseSqrtDiag * inverseSqrtDiag);
      
//      cout << "expected: " << expected << "; actual: " << actual << endl;
      
//      cout << "normalDiagValue: " << normalDiagValue << "; 1.0 / (inverseSqrtDiag * inverseSqrtDiag): ";
//      cout << 1.0 / (inverseSqrtDiag * inverseSqrtDiag) << endl;
      
      double err = abs(expected - actual);
      if (err > tol)
      {
        success = false;
      }
      maxErr = max(maxErr, err);
    }
    if (!success)
    {
      out << "max difference between expected and actual values was " << maxErr << endl;
    }
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonMatricesAgree )
  {
    out << "PoissonMatricesAgree test unimplemented!  Setting success = false.\n";
    success = false;
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonRHSesAgree)
  {
    out << "PoissonRHSesAgree test unimplemented!  Setting success = false.\n";
    success = false;
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonSolutionsAgree )
  {
    // tests that the solution we get out from DLS agrees with that we get from the standard solve.
    out << "PoissonSolutionsAgree test unimplemented!  Setting success = false.\n";
    success = false;
  }

} // namespace
