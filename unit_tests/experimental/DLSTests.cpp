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
#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"
#include "TpetraExt_MatrixMatrix_decl.hpp"

using namespace Camellia;
using namespace Teuchos;
using namespace std;

namespace
{
  typedef Teuchos::RCP<DLS<double>> DLSPtr;
  void setupSolve(SolutionPtr &standard, DLSPtr &dls, int H1Order = 2, int testSpaceEnrichment = 2, bool skipBCs = false)
  {
    int spaceDim = 2;
    bool useConformingTraces = true;
    int meshWidth = 2;
    double domainWidth = 1.0;
    
    vector<double> dimensions(spaceDim, domainWidth);
    vector<int> elementWidths(spaceDim,meshWidth);
    
    PoissonFormulation form(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dimensions, elementWidths,
                                                H1Order, testSpaceEnrichment);

    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(1.0 * form.v());
    
    BCPtr bc = BC::bc();
    if (!skipBCs)
    {
      bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), Function::zero());
    }
    
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
    {
      auto comm = MPIWrapper::CommWorld();
      int numProc = comm->NumProc();
      if (numProc > 1)
      {
        // for reasons unknown, we hang if numProc > 1
        cout << "skipping DLSTest PoissonDiagonalsAgree because numProc > 1\n";
        success = false;
        return;
      }
    }
    
    DLSPtr dls;
    SolutionPtr standardSolution;
    setupSolve(standardSolution, dls);
    
    Epetra_CommPtr comm = standardSolution->mesh()->Comm();
    
    TVectorPtr<double> dlsInverseSqrtDiag = dls->normalDiagInverseSqrt();
    
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
    /*
     Here, the basic notion is to test that the standard DPG system and the DLS system
     are related in the expected way -- namely, that the DPG system uses the normal matrix
     that arises from the DLS system.  The complication is that in order to implement DLS,
     we had to eliminate dofs on which we imposed BCs, which is not the way we handle BCs
     for the DPG system (there we impose them symmetrically, so we do come pretty close to
     eliminating them).  The upshot is this: if we take the DLS system and produce the
     corresponding normal matrix system, we expect to recover the DPG matrix, modulo some
     rows and columns that for the DPG system have only a unit entry in the diagonal.
     
     It's worth noting in this context that the global numbering between the DPG and DLS systems
     is the same -- the DLS system just lacks the GIDs on which BCs have been imposed.
     
     One other way in which the two systems disagree: we end up *scaling* the DLS matrix L
     in such a way that L'*L has unit diagonal.
     */
    
//    out << "PoissonMatricesAgree test unimplemented!  Setting success = false.\n";
//    success = false;
    
    {
      auto comm = MPIWrapper::CommWorld();
      int numProc = comm->NumProc();
      if (numProc > 1)
      {
        // for reasons unknown, we hang if numProc > 1
        cout << "skipping DLSTest PoissonMatricesAgree because numProc > 1\n";
        success = false;
        return;
      }
    }
    
    double tol=1e-12;
    double floor=1e-13;
    
    DLSPtr dls;
    SolutionPtr standardSolution;
    int H1Order = 2, testEnrichmentOrder = 2;
    bool skipBCs = true; // TODO: consider testing twice, once with BCs, and once without
    setupSolve(standardSolution, dls, H1Order, testEnrichmentOrder, skipBCs); // performs assembly

    auto mesh = standardSolution->mesh();
    
    TVectorPtr<double> dlsInverseSqrtDiag = dls->normalDiagInverseSqrt();
    TVector<double> dlsSqrtDiag = TVector<double>(dlsInverseSqrtDiag->getMap(),dlsInverseSqrtDiag->getNumVectors());
    dlsSqrtDiag.reciprocal(*dlsInverseSqrtDiag);
    
    // DEBUGGING; this won't look right on anything but a serial run...
//    cout << "dlsInverseSqrtDiag = " << dlsSqrtDiag;
//    auto diagValues_2d = dlsInverseSqrtDiag->template getLocalView<Kokkos::HostSpace> ();
//    // getLocalView returns a 2-D View by default.  We want a 1-D
//    // View, so we take a subview.
//    auto diagValues_1d = Kokkos::subview (diagValues_2d, Kokkos::ALL (), 0);
//    for (int localID=0; localID < diagValues_1d.size(); localID++)
//    {
//      cout << "dlsInverseSqrtDiag[" << localID << "] = " << diagValues_1d(localID) << endl;
//    }
    
    Epetra_CommPtr comm = mesh->Comm();
    
    auto K = standardSolution->getStiffnessMatrix();
    
    // if DLS matrix is L and standard stiffness matrix is K, then
    //       L' * L = D^1/2 * K * D^1/2
    
    auto L = dls->matrix();
    TMatrix<double> K_tpetra(L->getDomainMap(), 0);
    Tpetra::MatrixMatrix::Multiply(*L, true, *L, false, K_tpetra);
    
    using namespace std;
    cout << "L dimensions: " << L->getGlobalNumRows() << " x " << L->getGlobalNumCols() << endl;
    cout << "K_tpetra dimensions: " << K_tpetra.getGlobalNumRows() << " x " << K_tpetra.getGlobalNumCols() << endl;

    K_tpetra.rightScale(*dlsSqrtDiag.getVector(0));
    K_tpetra.leftScale (*dlsSqrtDiag.getVector(0));
    
    auto tpetraRowMap = K_tpetra.getRowMap();
    auto epetraRowMap = K->RowMap();
    auto tpetraColMap = K_tpetra.getColMap();
    auto epetraColMap = K->ColMap();
    auto minLocalRowIndex = tpetraRowMap->getMinLocalIndex();
    auto maxLocalRowIndex = tpetraRowMap->getMaxLocalIndex();
    auto maxEntryCount = K_tpetra.getNodeMaxNumRowEntries();
    vector<int> colIndexStorage(maxEntryCount); // local indices
    vector<double> valueStorage(maxEntryCount);
    vector<double> epetraValueStorage(maxEntryCount);
    vector<int> epetraColIndexStorage(maxEntryCount);
    for (int localRow=minLocalRowIndex; localRow <= maxLocalRowIndex; localRow++)
    {
      auto entryCount = K_tpetra.getNumEntriesInLocalRow(localRow);
      Teuchos::ArrayView<int> colIndices(&colIndexStorage[0], entryCount);
      Teuchos::ArrayView<double> values(&valueStorage[0], entryCount);
      
      K_tpetra.getLocalRowCopy(localRow, colIndices, values, entryCount);
      
      int epetraLocalRow = epetraRowMap.LID(tpetraRowMap->getGlobalElement(localRow));
      int epetraEntryCount = -1;
      K->ExtractMyRowCopy(epetraLocalRow, maxEntryCount, epetraEntryCount, &epetraValueStorage[0], &epetraColIndexStorage[0]);
      
      // compare:
      TEST_EQUALITY(epetraEntryCount, entryCount);
      
      if (epetraEntryCount == entryCount)
      {
        for (size_t entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
        {
          auto tpetraGlobalColIndex = tpetraColMap->getGlobalElement(colIndexStorage[entryOrdinal]);
          auto epetraGlobalColIndex = epetraColMap.GID(epetraColIndexStorage[entryOrdinal]);
          TEST_EQUALITY(tpetraGlobalColIndex, epetraGlobalColIndex);
          
          auto tpetraValue = valueStorage[entryOrdinal];
          auto epetraValue = epetraValueStorage[entryOrdinal];
          if ((std::abs(tpetraValue) > floor) || (std::abs(epetraValue) > floor))
          {
            TEST_FLOATING_EQUALITY(tpetraValue, epetraValue, tol);
          }
        }
      }
    }
    
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonRHSesAgree )
  {
    /*
     The DLS system is rectangular, test x trial.
     
     It has its BCs eliminated, and is scaled on the right by the normal matrix diagnonal.
     
     The usual DPG system is the normal matrix, constructed by computing Gram matrix G and rectangular
     (test x trial) bilinear form matrix B, then computing the optimal test coefficients:
    
     T = G \ B
    
     To test with these coefficients, we can compute the stiffness matrix K
     
     K = T^* B
     
     which is a square, trial x trial matrix.
     
     The load vector l_enriched on the enriched test space is similarly tested with 
     optimal test functions, l := T^* l_enriched.  (Length of l is trial.)
     
     By contrast, in the DLS system, we compute the Cholesky factorization of G
     
     L^* L = G
     
     And then define
     
     T_L = L \ B
     
     and
     
     D_L = diag( T_L^* T_L ) = diag( K )
     
     The system we seek to solve is to find u_h minimizing the residual
     
     || T_L * u_h - L \ l_enriched ||
     
     where the norm is is "little l^2" norm.  We also want to scale this system,
     somehow, so that the diagonal of the corresponding normal system is unity.
     
     I think this does *not* involve scaling the RHS, but instead involves scaling the
     solution u_h by the square root of the scaling matrix.
     
     To use the standard DPG solver to test the DLS RHS, we need to do a few things:
     - compute l_enriched
     - compute G \ l_enriched
     - compute L^T * G \ l_enriched = L \ l_enriched
     
     The last should match the RHS for the DLS system.
     
     */
    
    // The below code is adapted from BF::factoredCholeskySolve, which takes as arguments a Gram matrix ipMatrix, an "enriched" stiffness matrix
    // stiffnessEnriched, and an enriched load, rhsEnriched, and returns a normal stiffness and corresponding load.
    // We want to modify this to just compute the load with the Cholesky factorization (L^* \) applied.
    //

    {
      auto comm = MPIWrapper::CommWorld();
      int numProc = comm->NumProc();
      if (numProc > 1)
      {
        // for reasons unknown, we hang if numProc > 1
        cout << "skipping DLSTest because numProc > 1\n";
        success = false;
        return;
      }
    }
    
    
    DLSPtr dls;
    SolutionPtr standardSolution;
    // skip BCs to simplify this test (so that no RHS dofs will be eliminated)
    int H1Order = 2, testEnrichmentOrder = 2;
    bool skipBCs = true;
    setupSolve(standardSolution, dls, H1Order, testEnrichmentOrder, skipBCs);
    auto rhsVector = dls->rhs();
  
    auto mesh = standardSolution->mesh();
    auto rhs = standardSolution->rhs();
    
    TVectorPtr<double> dlsInverseSqrtDiag = dls->normalDiagInverseSqrt();
    
    Epetra_CommPtr comm = mesh->Comm();
    
    GlobalIndexType standardSolutionDofCount = standardSolution->getDofInterpreter()->globalDofCount();
    GlobalIndexType indexBase = 0;
    // we'll replicate standard solution diagonal on every rank:
    Epetra_LocalMap localMap(standardSolutionDofCount, indexBase, *comm);
    Epetra_Vector normalDiag(localMap);
    standardSolution->getStiffnessMatrix()->ExtractDiagonalCopy(normalDiag);
    
    // compare:
    using namespace Tpetra;
    typedef Map<IndexType,GlobalIndexType> Map;
    typedef RCP<const Map> constMapRCP;
    constMapRCP dlsMap = dlsInverseSqrtDiag->getMap();
    
    int myTestIDOffset = 0;
    int myTestDofCount = 0;
    auto myCells = standardSolution->mesh()->cellIDsInPartition();
    for ( GlobalIndexType cellID : myCells )
    {
      auto elemType = standardSolution->mesh()->getElementType(cellID);
      DofOrderingPtr testDofOrdering  = elemType->testOrderPtr;
      int numTestDofs = testDofOrdering->totalDofs(); // on each cell
      myTestDofCount += numTestDofs;
    }
    comm->ScanSum(&myTestDofCount, &myTestIDOffset, 1);
    
    myTestIDOffset -= myTestDofCount;
    auto entryOffset = 0; // into the local rhs vector
    auto localVectorView = rhsVector->get1dViewNonConst();
    
    for ( GlobalIndexType cellID : myCells )
    {
      auto elemType = standardSolution->mesh()->getElementType(cellID);
      DofOrderingPtr testDofOrdering  = elemType->testOrderPtr;
      int numTestDofs = testDofOrdering->totalDofs(); // on each cell
//      cout << "numTestDofs = " << numTestDofs << endl;
    
      using namespace Intrepid;
      int numCells = 1;
      FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
    
      auto ip = standardSolution->ip();
      bool testVsTest = true;
      auto basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);
      ip->computeInnerProductMatrix(ipMatrix, testDofOrdering, basisCache);
    
      int N = numTestDofs;
      
      char UPLO = 'L'; // lower-triangular
      
      int result = 0;
      int INFO;
      
      Teuchos::LAPACK<int, double> lapack;
      Teuchos::BLAS<int, double> blas;
      
      // DPOEQU( N, A, LDA, S, SCOND, AMAX, INFO )
      FieldContainer<double> scaleFactors(N);
      double scond, amax;
      lapack.POEQU(N, &ipMatrix[0], N, &scaleFactors[0], &scond, &amax, &INFO);
      
//      cout << "scaleFactors:\n" << scaleFactors;
      
      // do we need to equilibriate?
      // for now, we don't check, but just do the scaling...
      for (int i=0; i<N; i++)
      {
        double scale_i = scaleFactors[i];
        for (int j=0; j<N; j++)
        {
          ipMatrix(i,j) *= scale_i * scaleFactors[j];
        }
      }
      
//      cout << "ipMatrix equilibriated:\n" << ipMatrix;
      
      bool equilibriated = true;
      
      lapack.POTRF(UPLO, N, &ipMatrix[0], N, &INFO);
      
      if (INFO != 0)
      {
        cout << "dpotrf_ result: " << INFO << endl;
        result = INFO;
      }
      
      if (equilibriated)
      {
        // unequilibriate in the L factors:
        for (int j=0; j<N; j++)
        {
          for (int i=j; i<N; i++)
          {
            // lower-triangle is stored in (i,j) where i >= j
            ipMatrix(j,i) /= scaleFactors[i]; // FieldContainer transposes, effectively
          }
        }
      }

      //    cout << "ipMatrix, factored (lower-tri):\n" << ipMatrix;

      // skip over the part that deals with the stiffness matrix
      double ALPHA = 1.0;
//      blas.TRSM(Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, N, M, ALPHA, &ipMatrix[0], N,
//                &stiffnessEnriched[0], N);
//      
//      //    cout << "stiffnessEnriched, back-subbed:\n" << stiffnessEnriched;
//      
//      double BETA = 0.0;
//      blas.SYRK(Teuchos::LOWER_TRI, Teuchos::TRANS, M, N, ALPHA, &stiffnessEnriched[0], N, BETA, &stiffness[0], M);

      // copy lower-triangular part of stiffness to the upper-triangular part (in column-major/Fortran order)
//      for (int i=0; i<M; i++)
//      {
//        for (int j=i+1; j<M; j++)
//        {
//          // lower-triangle is stored in (i,j) where j >= i
//          // column-major: store (i,j) in i*M+j
//          // set (j,i) := (i,j)
//          stiffness[j*M+i] = stiffness[i*M+j];
//        }
//      }

      //    cout << "stiffness matrix:\n" << stiffness;
      
      int oneColumn = 1;
      
      FieldContainer<double> rhsEnriched(numTestDofs);
      rhs->integrateAgainstStandardBasis(rhsEnriched, testDofOrdering, basisCache);
      
      blas.TRSM(Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, N, oneColumn,
                ALPHA, &ipMatrix[0], N, &rhsEnriched[0], N);
      
      // what's in there is L \ rhs_enriched, which is the rhs we expect, modulo node numbering (which is simple, since
      // there is no overlap in the test space) and imposition of BCs, which we have bracketed by nullifying the BCs.
      
      double tol = 1e-15;
      double floor = tol * 10;
      for (int testOrdinal=0; testOrdinal<numTestDofs; testOrdinal++)
      {
        double expectedValue = rhsEnriched[testOrdinal];
        double actualValue = localVectorView[entryOffset+testOrdinal];
        if ((abs(expectedValue) > floor) || (abs(actualValue) > floor))
        {
          TEUCHOS_TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol, out, success);
        }
        else
        {
          out << "abs(" << expectedValue << ") < " << floor << "; ";
          out << "abs(" << actualValue << ") < " << floor << " -- skipping floating equality test\n";
        }
      }
      
      entryOffset += numTestDofs;
    }
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonBCImposition )
  {
    // not too sure what I want to test here, just that it would be nice to have something that granularly tests
    // the imposition of BCs.
    out << "PoissonBCImposition test unimplemented!  Setting success = false.\n";
    success = false;
  }
  
  TEUCHOS_UNIT_TEST( DLS, PoissonSolutionsAgree )
  {
    // tests that the solution we get out from DLS agrees with that we get from the standard solve.
    out << "PoissonSolutionsAgree test unimplemented!  Setting success = false.\n";
    success = false;
  }

} // namespace
