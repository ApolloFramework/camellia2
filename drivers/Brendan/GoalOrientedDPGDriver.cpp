//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  PoissonGoalOrientedDPGDriver.cpp
//  Driver for goal-oriented adaptive mesh refinement for Poisson's BVP
//  Camellia
//
//  Created by Brendan Keith, December 2017.

#include "math.h"

#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "ErrorIndicator.h"
#include "GMGSolver.h"
#include "H1ProjectionFormulation.h"
#include "PoissonFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
// #include "PreviousSolutionFunction.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"

#include "CamelliaDebugUtility.h"


using namespace Camellia;


int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options



  //////////////////////////////////////////////////////////////////////
  ///////////////////////  COMMAND LINE PARAMETERS  ////////////////////
  //////////////////////////////////////////////////////////////////////
  string formulation = "Poisson";
  string problemChoice = "SquareDomain";
  string formulationChoice = "ULTRAWEAK";
  int spaceDim = 2;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  int numXElems = 2;
  int numYElems = 2;
  string norm = "Graph";
  string errorIndicator = "Uniform";
  bool useConformingTraces = true;
  string solverChoice = "KLU";
  string multigridStrategyString = "V-cycle";
  bool useCondensedSolve = false;
  bool useConjugateGradient = true;
  bool logFineOperator = false;
  double solverTolerance = 1e-8;
  int maxLinearIterations = 1000;
  // bool computeL2Error = false;
  bool exportSolution = false;
  string outputDir = ".";
  string tag="";
  cmdp.setOption("formulation", &formulation, "Poisson");
  cmdp.setOption("problem", &problemChoice, "SquareDomain");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("formulationChoice", &formulationChoice, "ULTRAWEAK");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variables");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("numYElems",&numYElems,"number of elements in y direction");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy,Uniform,GoalOrientedOden");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);



  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr n    = Function::normal();
  const static double PI  = 3.141592653589793238462;
  FunctionPtr sin_pix = Teuchos::rcp( new Sin_ax(PI) );
  FunctionPtr sin_piy = Teuchos::rcp( new Sin_ay(PI) );
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );



  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////

  //////////////////////  DECLARE MESH TOPOLOGY  ///////////////////////
  // MeshGeometryPtr spatialMeshGeom;
  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;
  if (problemChoice == "SquareDomain")
  {
    double x0 = 0.0, y0 = 0.0;
    double width = 1.0;
    double height = 1.0;
    int horizontalCells = 2, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  ///////////////////////  DECLARE BILINEAR FORM  //////////////////////
  bool useEnrichedTraces = true;
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);  // does this need to be here?
  // double lengthScale = 1.0;
  // H1ProjectionFormulation form(spaceDim, useConformingTraces, H1ProjectionFormulation::CONTINUOUS_GALERKIN, lengthScale);
  // if (formulationChoice == "ULTRAWEAK")
  //   PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::ULTRAWEAK);
  // else
  // {
  //   cout << "ERROR: formulationChoice not currently supported. Returning null.\n";
  //   return Teuchos::null;
  // }
  PoissonFormulation form(spaceDim, useConformingTraces);

  BFPtr bf = form.bf();
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);


  ///////////////////////////  DECLARE MESH  ///////////////////////////
  vector<int> H1Order = {k + 1};
  int testEnrichment = delta_k;
  MeshPtr mesh = Teuchos::rcp( new Mesh(spatialMeshTopo, bf, H1Order, testEnrichment) ) ;
  if (globalEdgeToCurveMap.size() > 0) // only necessary if geometry is curvilinear
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }


  //////////////////////  DECLARE TRIAL FUNCTIONS //////////////////////
  VarPtr u, sigma, u_hat, sigma_n_hat;
  u           = form.u();
  sigma       = form.sigma();
  u_hat       = form.u_hat(); 
  sigma_n_hat = form.sigma_n_hat();


  //////////////////////  DECLARE TEST FUNCTIONS ///////////////////////
  VarPtr v, tau;
  v   = form.v();
  tau = form.tau();


  ///////////////////  DECLARE RHS & EXACT SOLUTION  ///////////////////
  FunctionPtr u_exact, f;
  // u_exact = one;
  // u_exact = x * x + 2 * x * y;
  // u_exact = x * x * x * y + 2 * x * y * y;
  u_exact = sin_pix * sin_piy;
  f = u_exact->dx()->dx() + u_exact->dy()->dy();
  RHSPtr rhs = form.rhs(f);


  ////////////////////  DECLARE BOUNDARY CONDITIONS ////////////////////
  BCPtr bc = BC::bc();
  if (problemChoice == "SquareDomain")
  {
    SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
    SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
    SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
    SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
    bc->addDirichlet(u_hat, x_equals_zero, u_exact);
    bc->addDirichlet(u_hat, y_equals_zero, u_exact);
    bc->addDirichlet(u_hat, x_equals_one, u_exact);
    bc->addDirichlet(u_hat, y_equals_one, u_exact);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  ///////////////////////  DECLARE INNER PRODUCT ///////////////////////
  map<string, IPPtr> poissonIPs;
  poissonIPs["Graph"] = bf->graphNorm();
  IPPtr ip = poissonIPs[norm];


  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  // soln->setBC(bc);


  //////////////////////////////////////////////////////////////////////
  ///////////////////////////////  SOLVE  //////////////////////////////
  //////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Teuchos::Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");

  if (commRank == 0)
    Solver::printAvailableSolversReport();
  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#ifdef HAVE_AMESOS_MUMPS
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
  int azOutput = 50; // print residual every 20 CG iterations

  GMGOperator::MultigridStrategy multigridStrategy;
  if (multigridStrategyString == "Two-level")
  {
    multigridStrategy = GMGOperator::TWO_LEVEL;
  }
  else if (multigridStrategyString == "W-cycle")
  {
    multigridStrategy = GMGOperator::W_CYCLE;
  }
  else if (multigridStrategyString == "V-cycle")
  {
    multigridStrategy = GMGOperator::V_CYCLE;
  }
  else if (multigridStrategyString == "Full-V")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_V;
  }
  else if (multigridStrategyString == "Full-W")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_W;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized multigrid strategy");
  }

  ostringstream solnName;
  solnName << "GoalOrientedDPG" << "_" << norm << "_k" << k << "_" << solverChoice;
  if (solverChoice[0] == 'G')
    solnName << "_" << multigridStrategyString;
  if (tag != "")
    solnName << "_" << tag;
  solnName << "_" << formulation;

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";  

  ofstream dataFile(dataFileLocation);
  dataFile << "Ref\t "
           << "Elements\t "
           << "DOFs\t "
           << "Energy\t "
           << "Rate\t "
           << "Solvetime\t "
           << "Elapsed\t "
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));

  double energyThreshold = 0.5;
  RefinementStrategyPtr refStrategy;
  if (errorIndicator == "Uniform")
  {
    energyThreshold = 0;
    refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
  }
  else if (errorIndicator == "Energy")
  {
    refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized refinement strategy");
  }

  SolverPtr solver;
  double energyErrorPrvs, numGlobalDofsPrvs;
  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
    Teuchos::RCP<GMGSolver> gmgSolver;
    if (solverChoice[0] == 'G')
    {
      bool reuseFactorization = true;
      SolverPtr coarseSolver = Solver::getDirectSolver(reuseFactorization);
      int kCoarse = 1;
      vector<MeshPtr> meshSequence = GMGSolver::meshesForMultigrid(mesh, kCoarse, delta_k);
      // for (int i=0; i < meshSequence.size(); i++)
      // {
      //   if (commRank == 0)
      //     cout << meshSequence[i]->numGlobalDofs() << endl;
      // }
      while (meshSequence[0]->numGlobalDofs() < 100000 && meshSequence.size() > 2)
        meshSequence.erase(meshSequence.begin());
      gmgSolver = Teuchos::rcp(new GMGSolver(soln, meshSequence, maxLinearIterations, solverTolerance, multigridStrategy, coarseSolver, useCondensedSolve));
      gmgSolver->setUseConjugateGradient(useConjugateGradient);
      int azOutput = 20; // print residual every 20 CG iterations
      gmgSolver->setAztecOutput(azOutput);
      gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");

      if (solverChoice == "GMG-Direct")
        gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::Direct);
      if (solverChoice == "GMG-ILU")
        gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::ILU);
      if (solverChoice == "GMG-IC")
        gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::IC);
      solver = gmgSolver;
    }
    else if (refIndex == 0)
      solver = solvers[solverChoice];

    solverTime->start(true);

    soln->solve(solver);

    double solveTime = solverTime->stop();

    double energyError = soln->energyErrorTotal();

    double numGlobalDofs = mesh->numGlobalDofs();

    double energyRate;
    if (refIndex == 0)
      energyRate = 0;
    else
    {
      energyRate =-spaceDim*log(energyErrorPrvs/energyError)/log(numGlobalDofsPrvs/numGlobalDofs);
    }

    if (commRank == 0)
    {
      cout << setprecision(8) 
        << " \nRefinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << " \tEnergy Error: " << energyError
        << setprecision(4)
        << " \tRate: " << energyRate
        << " \nSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << endl;
      dataFile << setprecision(8)
        << " " << refIndex
        << " " << mesh->numActiveElements()
        << " " << numGlobalDofs
        << " " << energyError
        << setprecision(4)
        << " " << energyRate
        << " " << solveTime
        << " " << totalTimer->totalElapsedTime(true)
        << endl;
    }

    energyErrorPrvs = energyError;
    numGlobalDofsPrvs = numGlobalDofs;

    if (exportSolution)
    {
      exporter->exportSolution(soln, refIndex);
      
      // // output mesh with GnuPlotUtil
      // ostringstream meshExportName;
      // meshExportName << outputDir << "/" << solnName.str() << "/" << "ref" << refIndex << "_mesh";
      // // int numPointsPerEdge = 3;
      // bool labelCells = false;
      // string meshColor = "black";
      // GnuPlotUtil::writeComputationalMeshSkeleton(meshExportName.str(), mesh, labelCells, meshColor);
    }

    if (refIndex != numRefs)
    {
      ///////////////////  CHOOSE REFINEMENT STRATEGY  ////////////////////
      // Can also set whether to use h or p adaptivity here
      refStrategy->refine();
    } 

  }

  // dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
