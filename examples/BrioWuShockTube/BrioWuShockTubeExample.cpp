//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "ExpFunction.h" // defines Ln
#include "Function.h"
#include "Functions.hpp"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "IdealMHDFormulation.hpp"
#include "LagrangeConstraints.h"
#include "RHS.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

void addConservationConstraint(Teuchos::RCP<IdealMHDFormulation> form)
{
  int rank = MPIWrapper::CommWorld()->MyPID();
  cout << "TRYING CONSERVATION ENFORCEMENT.\n";
  auto soln = form->solution();
  auto solnIncrement = form->solutionIncrement();
  auto prevSoln = form->solutionPreviousTimeStep();
  auto bf  = solnIncrement->bf();
  auto rhs = solnIncrement->rhs();
  auto dt = form->getTimeStep();
  
  Teuchos::RCP<LagrangeConstraints> constraints = Teuchos::rcp(new LagrangeConstraints);
  // vc constraint:
  VarPtr vc = form->vc();
  map<int, FunctionPtr> vcEqualsOne = {{vc->ID(), Function::constant(1.0)}};
  LinearTermPtr vcTrialFunctional = bf->trialFunctional(vcEqualsOne) * dt;
  FunctionPtr   vcRHSFunction     = rhs->linearTerm()->evaluate(vcEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(vcTrialFunctional == vcRHSFunction);
  if (rank == 0) cout << "Added element constraint " << vcTrialFunctional->displayString() << " == " << vcRHSFunction->displayString() << endl;

  const int spaceDim = 1;
  // vm constraint(s):
  for (int d=0; d<spaceDim; d++)
  {
    // test with 1
    VarPtr vm = form->vm(d+1);
    map<int, FunctionPtr> vmEqualsOne = {{vm->ID(), Function::constant(1.0)}};
    LinearTermPtr trialFunctional = bf->trialFunctional(vmEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
    FunctionPtr rhsFxn = rhs->linearTerm()->evaluate(vmEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
    constraints->addConstraint(trialFunctional == rhsFxn);

    if (rank == 0) cout << "Added element constraint " << trialFunctional->displayString() << " == " << rhsFxn->displayString() << endl;
  }
  // ve constraint:
  VarPtr ve = form->ve();
  map<int, FunctionPtr> veEqualsOne = {{ve->ID(), Function::constant(1.0)}};
  LinearTermPtr veTrialFunctional = bf->trialFunctional(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  FunctionPtr   veRHSFunction     = rhs->linearTerm()->evaluate(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(veTrialFunctional == veRHSFunction);
  if (rank == 0) cout << "Added element constraint " << veTrialFunctional->displayString() << " == " << veRHSFunction->displayString() << endl;

  // although enforcement only happens in solnIncrement, the constraints change numbering of dofs, so we need to set constraints in each Solution object
  solnIncrement->setLagrangeConstraints(constraints);
  soln->setLagrangeConstraints(constraints);
  prevSoln->setLagrangeConstraints(constraints);
}

void writeFunctions(MeshPtr mesh, int meshWidth, int polyOrder, double x_a, std::vector<FunctionPtr> &functions, std::vector<string> &functionNames, string filePrefix)
{
  int numPoints = max(polyOrder*2 + 1, 2); // polyOrder * 2 will, hopefully, give reasonable approximation of high-order curves
  Intrepid::FieldContainer<double> refCellPoints(numPoints,1);
  double dx = 2.0 / (refCellPoints.size() - 1); // 2.0 is the size of the reference element
  for (int i=0; i<refCellPoints.size(); i++)
  {
    refCellPoints(i,0) = -1.0 + dx * i;
  }
//  cout << "numPoints = " << numPoints << endl;
//  cout << "dx = " << dx << endl;
//  cout << "refCellPoints:\n" << refCellPoints;
  
  int numFunctions = functionNames.size();
  vector<Intrepid::FieldContainer<double>> pointData(numFunctions, Intrepid::FieldContainer<double>(meshWidth,numPoints));
  for (int functionOrdinal=0; functionOrdinal<numFunctions; functionOrdinal++)
  {
    pointData[functionOrdinal].initialize(0.0);
  }
  Teuchos::Array<int> cellDim;
  cellDim.push_back(1);  // C
  cellDim.push_back(numPoints); // P
  auto & myCellIDs = mesh->cellIDsInPartition();
  for (auto cellID : myCellIDs)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    basisCache->setRefCellPoints(refCellPoints);
    
    for (int functionOrdinal=0; functionOrdinal<numFunctions; functionOrdinal++)
    {
      auto f = functions[functionOrdinal];
      auto name = functionNames[functionOrdinal];
      
      Intrepid::FieldContainer<double> localData(cellDim, &pointData[functionOrdinal](cellID,0));
      
      f->values(localData, basisCache);
    }
  }
  
  for (auto & data : pointData)
  {
    MPIWrapper::entryWiseSum(*mesh->Comm(), data);
  }
  
  double h = 1.0 / meshWidth;
  for (int functionOrdinal=0; functionOrdinal<functionNames.size(); functionOrdinal++)
  {
    int totalPoints = meshWidth*(numPoints-1); // cells overlap; eliminate duplicates
    Intrepid::FieldContainer<double> xyPoints(totalPoints,2);
    
    ostringstream fileName;
    fileName << filePrefix << "_" << functionNames[functionOrdinal] << ".dat";
    
    ofstream fout(fileName.str());
    fout << setprecision(6);
    
    for (int i=0; i<totalPoints; i++)
    {
      double x = x_a + dx * i / 2.0 * h;
      int cellOrdinal  = i / (numPoints-1);
      int pointOrdinal = i % (numPoints-1);
      if (i == totalPoints - 1)
      {
        // for last cell, we do take its final point...
        pointOrdinal = numPoints - 1;
      }
      
      double y = pointData[functionOrdinal](cellOrdinal,pointOrdinal);
      xyPoints(i,0) = x;
      xyPoints(i,1) = y;
    }
    GnuPlotUtil::writeXYPoints(fileName.str(), xyPoints);
  }
}

enum TestNormChoice
{
  STEADY_GRAPH_NORM,
  TRANSIENT_GRAPH_NORM,
  EXPERIMENTAL_CONSERVATIVE_NORM
};

FunctionPtr ln(FunctionPtr arg)
{
  return Teuchos::rcp(new Ln<double>(arg));
}

template<class Form>
int runSolver(Teuchos::RCP<Form> form, double dt, int meshWidth, double x_a, double x_b,
              int polyOrder, int cubatureEnrichment, bool useCondensedSolve, double nonlinearTolerance,
              TestNormChoice normChoice, bool enforceConservationUsingLagrangeMultipliers,
              bool runSodInstead, bool spaceTime)
{
  MeshPtr mesh = form->solutionIncrement()->mesh();
  
  if (enforceConservationUsingLagrangeMultipliers)
  {
    addConservationConstraint(form);
  }
  int rank = Teuchos::GlobalMPISession::getRank();
  
  SolutionPtr solnIncrement = form->solutionIncrement();
  SolutionPtr soln = form->solution();
  SolutionPtr solnPreviousTime = form->solutionPreviousTimeStep();
  
  auto ip = solnIncrement->ip(); // this will be the transient graph norm...
  if (normChoice == STEADY_GRAPH_NORM)
  {
    auto steadyBF = form->steadyBF();
    auto steadyIP = steadyBF->graphNorm();
    auto bf = form->bf();
    
    // steadyBF will have no rho terms wherever velocity is zero
    // unclear what the right thing to do is; for now we just add something in
    steadyBF = Teuchos::rcp( new BF(*steadyBF) ); // default copy constructor
    steadyBF->addTerm(form->rho(),form->vc());
    
    cout << "steadyBF: " << steadyBF->displayString() << endl;
    bf->setBFForOptimalTestSolve(steadyBF);
    cout << "Using steady graph norm:\n";
    steadyIP->printInteractions();
    soln->setIP(steadyIP);
    solnIncrement->setIP(steadyIP);
    form->solutionPreviousTimeStep()->setIP(steadyIP);
  }
  else if (normChoice == EXPERIMENTAL_CONSERVATIVE_NORM)
  {
    auto dt = form->getTimeStep();
    auto steadyIP = IP::ip();
    // let's walk through the members, dropping any summands that involve test terms without gradients
    auto linearTerms = ip->getLinearTerms();
    for (auto lt : linearTerms)
    {
      LinearTermPtr revisedLT = Teuchos::rcp(new LinearTerm);
      auto summands = lt->summands();
      for (auto summand : summands)
      {
        auto weight = summand.first;
        auto var = summand.second;
        if (var->op() == OP_VALUE)
        {
          continue; // skip this summand
        }
        else
        {
          revisedLT = revisedLT + weight * var;
        }
      }
      if (revisedLT->summands().size() > 0)
      {
        steadyIP->addTerm(revisedLT);
      }
    }
    // now, add boundary terms for each test function:
    steadyIP->addBoundaryTerm(form->vc());
    steadyIP->addBoundaryTerm(form->vm(1));
    steadyIP->addBoundaryTerm(form->ve());
    cout << "Using modified steady graph norm:\n";
    steadyIP->printInteractions();
    soln->setIP(steadyIP);
    solnIncrement->setIP(steadyIP);
    form->solutionPreviousTimeStep()->setIP(steadyIP);
  }
  
  double finalTime = 0.08;
  if (runSodInstead)
  {
    finalTime = 0.20;
  }
  int numTimeSteps = spaceTime ? 1 : finalTime / dt; // run simulation to t = 0.08
  
  const double By_FACTOR = .01; // Should be 1.0.  Trying something else to see if we can solve an easier problem.
  
  if (rank == 0)
  {
    using namespace std;
    cout << "Solving with:\n";
    cout << "p  = " << polyOrder << endl;
    cout << "dt = " << dt << endl;
    if (spaceTime)
    {
      int totalElements = mesh->numActiveElements();
      int temporalElements = totalElements / meshWidth;
      cout << meshWidth << " spatial elements; " << temporalElements << " temporal divisions.\n";
    }
    else
    {
      cout << meshWidth << " elements; " << numTimeSteps << " timesteps.\n";
    }
    if (By_FACTOR != 1.0)
    {
      cout << "NOTE: SOLVING MODIFIED PROBLEM: multiplying By initial conditions by " << By_FACTOR << endl;
    }
  }
  
  form->setTimeStep(dt);
  solnIncrement->setUseCondensedSolve(useCondensedSolve);
  solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichment);
//  solnIncrement->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
//  solnIncrement->setWriteRHSToMatrixMarketFile(   true, "/tmp/b.dat");
  
  double gamma = form->gamma();
  double c_v   = form->Cv();
  
  double rho_a = 1.0; // prescribed density at left
  double p_a   = 1.0; // prescribed pressure at left
  double By_a  = 1.0 * By_FACTOR;
  double Bz_a  = 0.0;
  
  double rho_b = 0.125;
  double p_b   = 0.1;
  double By_b  = -1.0 * By_FACTOR;
  double Bz_b  = 0.0;
  
  double Bx_a = 0.75;
  double Bx_b = 0.75;
  
  auto B_a = Function::vectorize(Function::constant(Bx_a), Function::constant(By_a), Function::constant(Bz_a));
  auto B_b = Function::vectorize(Function::constant(Bx_b), Function::constant(By_b), Function::constant(Bz_b));
  
  if (runSodInstead)
  {
    Bx_a = 0.0;
    Bx_b = 0.0;
    By_a = 0.0;
    By_b = 0.0;
  }
  
  double T_a   = p_a / (rho_a * (gamma - 1.) * c_v);
  double T_b   = p_b / (rho_b * (gamma - 1.) * c_v);
  
  double E_a   = c_v * rho_a * T_a; // + 0.5 * u dot u [ == 0]
  double E_b   = c_v * rho_b * T_b; // + 0.5 * u dot u [ == 0]
  
  double R  = form->R();
  double Cv = form->Cv();
  
  if (rank == 0)
  {
    cout << "R =     " << R << endl;
    cout << "Cv =    " << Cv << endl;
    cout << "gamma = " << gamma << endl;
    cout << "State on left:\n";
    cout << "rho = " << rho_a << endl;
    cout << "p   = " << p_a   << endl;
    cout << "E   = " << E_a   << endl;
    cout << "Bx  = " << Bx_a   << endl;
    cout << "By  = " << By_a   << endl;
    cout << "Bz  = " << Bz_a   << endl;
    cout << "State on right:\n";
    cout << "rho = " << rho_b << endl;
    cout << "p   = " << p_b   << endl;
    cout << "E   = " << E_b   << endl;
    cout << "Bx  = " << Bx_b   << endl;
    cout << "By  = " << By_b   << endl;
    cout << "Bz  = " << Bz_b   << endl;
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  {
    auto H_right = Function::heaviside((x_a + x_b)/2.0); // Heaviside is 0 left of center, 1 right of center
    auto H_left  = 1.0 - H_right;  // this guy is 1 left of center, 0 right of center
    auto step = [&](double val_a, double val_b)
    {
      return H_left * val_a + H_right * val_b;
    };
    
    FunctionPtr rho = step(rho_a,rho_b);
    FunctionPtr E   = step(E_a, E_b);
    
    // for Ideal MHD, even in 1D, u is a vector, as is B
    auto zero = Function::zero();
    auto velocityVector = Function::vectorize(zero, zero, zero);
    
    auto Bx = runSodInstead ? zero : Function::constant(0.75);
    auto By = step(By_a, By_b);
    auto Bz = step(Bz_a, Bz_a);
    auto BVector = Function::vectorize(Bx, By, Bz);
    
    form->setInitialCondition(rho, velocityVector, E, BVector);
    if (spaceTime)
    {
      // have had some issues using the discontinuous initial guess for all time in Sod problem at least
      // let's try something much more modest: just unit values for rho, E, B, zero for u
      FunctionPtr one    = Function::constant(1.0);
      FunctionPtr rhoOne = one;
      FunctionPtr EOne   = one;
      FunctionPtr BGuess;
      if (!runSodInstead)
      {
        BGuess   = Function::vectorize(Bx, one, one);
      }
      else
      {
        BGuess   = Function::vectorize(zero, zero, zero);
      }
      auto initialGuess = form->exactSolutionFieldMap(rhoOne, velocityVector, EOne, BGuess);
      form->setInitialState(initialGuess);
    }
  }
  
  auto & prevSolnMap = form->solutionPreviousTimeStepFieldMap();
  auto & solnMap     = form->solutionFieldMap();
  auto pAbstract = form->abstractPressure();
  auto TAbstract = form->abstractTemperature();
  auto uAbstract = form->abstractVelocity()->x();
  auto vAbstract = form->abstractVelocity()->y();
  auto wAbstract = form->abstractVelocity()->z();
  auto mAbstract = form->abstractMomentum();
  auto EAbstract = form->abstractEnergy();
  
  // define the pressure so we can plot in our solution export
  FunctionPtr p_prev = pAbstract->evaluateAt(prevSolnMap);
  FunctionPtr p_soln = pAbstract->evaluateAt(solnMap);
  
  // define u so we can plot in solution export
//  cout << "uAbstract = " << uAbstract->displayString() << endl;
  FunctionPtr u_prev = uAbstract->evaluateAt(prevSolnMap);
  FunctionPtr v_prev = vAbstract->evaluateAt(prevSolnMap);
  
  auto velocityAbstract = form->abstractVelocity();
  auto velocity_soln = velocityAbstract->evaluateAt(solnMap);
  auto velocityMagnitude_soln = Function::sqrtFunction(dot(3,velocity_soln,velocity_soln));
  
  // define change in entropy so we can plot in our solution export
  // s2 - s1 = c_p ln (T2/T1) - R ln (p2/p1)
  FunctionPtr ds;
  {
    FunctionPtr p1 = p_prev;
    FunctionPtr p2 = pAbstract->evaluateAt(solnMap);
    FunctionPtr T1 = TAbstract->evaluateAt(prevSolnMap);
    FunctionPtr T2 = TAbstract->evaluateAt(solnMap);
    double c_p = form->Cp();
    ds = c_p * ln(T2 / T1) - R * ln(p2/p1);
  }
  
  if (spaceTime)
  {
    // DEBUGGING: print out all BF integrations (trying to see why we get zero rows)
//    solnIncrement->bf()->setPrintTermWiseIntegrationOutput(true);
  }
  
  vector<FunctionPtr> functionsToPlot = {p_prev,u_prev,ds};
  vector<string> functionNames = {"pressure","velocity","entropy change"};
  
  // history export gets every nonlinear increment as a separate step
  HDF5Exporter solutionHistoryExporter(mesh, "brioWuSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(mesh, "brioWuSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  solnName << "brioWuSolution" << "_dt" << dt << "_k" << polyOrder;
  HDF5Exporter solutionExporter(mesh, solnName.str(), ".");
  
  solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), 0.0);
  solutionHistoryExporter.exportSolution(form->solution(), 0.0);
  solutionExporter.exportSolution(form->solutionPreviousTimeStep(), functionsToPlot, functionNames, 0.0);
  
//  IPPtr naiveNorm = form->solutionIncrement()->bf()->naiveNorm(spaceDim);
//  if (rank == 0)
//  {
//    cout << "*************************************************************************\n";
//    cout << "**********************    USING NAIVE NORM    ***************************\n";
//    cout << "*************************************************************************\n";
//  }
//  form->solutionIncrement()->setIP(naiveNorm);
  
  SpatialFilterPtr leftX  = SpatialFilter::matchingX(x_a);
  SpatialFilterPtr rightX = SpatialFilter::matchingX(x_b);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  
  if (rank == 0)
  {
    std::cout << "E_a = " << E_a << std::endl;
    std::cout << "E_b = " << E_b << std::endl;
  }
  auto velocityVector = Function::vectorize(Function::zero(), Function::zero(), Function::zero());
  
  auto Bx = Function::constant(0.75);
  auto BVector_a = Function::vectorize(Bx, Function::constant(By_a), Function::constant(Bz_a));
  auto BVector_b = Function::vectorize(Bx, Function::constant(By_b), Function::constant(Bz_b));
  
  //  (SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
  form->addMassFluxCondition(          leftX, Function::constant(rho_a), velocityVector, Function::constant(E_a), BVector_a);
  form->addMassFluxCondition(         rightX, Function::constant(rho_b), velocityVector, Function::constant(E_b), BVector_b);
  form->addMomentumFluxCondition(      leftX, Function::constant(rho_a), velocityVector, Function::constant(E_a), BVector_a);
  form->addMomentumFluxCondition(     rightX, Function::constant(rho_b), velocityVector, Function::constant(E_b), BVector_b);
  form->addEnergyFluxCondition(        leftX, Function::constant(rho_a), velocityVector, Function::constant(E_a), BVector_a);
  form->addEnergyFluxCondition(       rightX, Function::constant(rho_b), velocityVector, Function::constant(E_b), BVector_b);
  form->addMagneticFluxCondition(      leftX, Function::constant(rho_a), velocityVector, Function::constant(E_a), BVector_a);
  form->addMagneticFluxCondition(     rightX, Function::constant(rho_b), velocityVector, Function::constant(E_b), BVector_b);
  
  FunctionPtr rho = Function::solution(form->rho(), form->solution());
  FunctionPtr m   = mAbstract->evaluateAt(solnMap);
  FunctionPtr E   = EAbstract->evaluateAt(solnMap);
  FunctionPtr tc  = Function::solution(form->tc(),  form->solution(), true);
  FunctionPtr tm  = Function::solution(form->tm(1), form->solution(), true);
  FunctionPtr te  = Function::solution(form->te(),  form->solution(), true);
  
  auto printConservationReport = [&]() -> void
  {
    double totalMass = rho->integrate(mesh);
    double totalXMomentum = m->spatialComponent(1)->integrate(mesh);
    double totalEnergy = E->integrate(mesh);
    double dsIntegral = ds->integrate(mesh);
    
    if (rank == 0)
    {
      cout << "Total Mass:        " << totalMass << endl;
      cout << "Total x Momentum:    " << totalXMomentum << endl;
      cout << "Total Energy:      " << totalEnergy << endl;
      cout << "Change in Entropy: " << dsIntegral << endl;
    }
  };
  
  // elementwise local momentum conservation:
  FunctionPtr rho_soln = Function::solution(form->rho(), form->solution());
  FunctionPtr rho_prev = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr rhoTimeStep = (rho_soln - rho_prev) / dt;
  FunctionPtr rhoFlux = Function::solution(form->tc(), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr momentum_soln = mAbstract->x()->evaluateAt(solnMap);
  FunctionPtr momentum_prev = mAbstract->x()->evaluateAt(prevSolnMap);
  FunctionPtr momentumTimeStep = (momentum_soln - momentum_prev) / dt;
  FunctionPtr momentumFlux = Function::solution(form->tm(1), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr energy_soln = EAbstract->evaluateAt(solnMap);
  FunctionPtr energy_prev = EAbstract->evaluateAt(prevSolnMap);
  FunctionPtr energyTimeStep = (energy_soln - energy_prev) / dt;
  FunctionPtr energyFlux = Function::solution(form->te(), form->solution(), true); // include sideParity weights

  vector<FunctionPtr> conservationFluxes = {rhoFlux,     momentumFlux,     energyFlux};
  vector<FunctionPtr> timeDifferences    = {rhoTimeStep, momentumTimeStep, energyTimeStep};
  
  int numConserved = conservationFluxes.size();
  
  auto printLocalConservationReport = [&]() -> void
  {
    auto & myCellIDs = mesh->cellIDsInPartition();
    int cellOrdinal = 0;
    vector<double> maxConservationFailures(numConserved);
    for (int conservedOrdinal=0; conservedOrdinal<numConserved; conservedOrdinal++)
    {
      double maxConservationFailure = 0.0;
      for (auto cellID : myCellIDs)
      {
//        cout << "timeDifferences function: " << timeDifferences[conservedOrdinal]->displayString() << endl;
        double timeDifferenceIntegral = timeDifferences[conservedOrdinal]->integrate(cellID, mesh);
        double fluxIntegral = conservationFluxes[conservedOrdinal]->integrate(cellID, mesh);
        double cellIntegral = timeDifferenceIntegral + fluxIntegral;
        maxConservationFailure = std::max(abs(cellIntegral), maxConservationFailure);
        cellOrdinal++;
      }
      maxConservationFailures[conservedOrdinal] = maxConservationFailure;
    }
    vector<double> globalMaxConservationFailures(numConserved);
    mesh->Comm()->MaxAll(&maxConservationFailures[0], &globalMaxConservationFailures[0], numConserved);
    if (rank == 0) {
      cout << "Max cellwise (rho,m,E) conservation failures: ";
      for (double failure : globalMaxConservationFailures)
      {
        cout << failure << "\t";
      }
      cout << endl;
    }
//    for (int cellOrdinal=0; cellOrdinal<myCellCount; cellOrdinal++)
//    {
//      cout << "cell ordinal " << cellOrdinal << ", conservation failure: " << cellIntegrals[cellOrdinal] << endl;
//    }
  };
  
//  {
//    // DEBUGGING
//    BFPtr bf = form->bf();
//    auto debugBF = Teuchos::rcp( new BF(bf->varFactory()) );
//    FunctionPtr dt = form->getTimeStep();
//    VarPtr m2 = form->m(2);
//    VarPtr vm2 = form->vm(2);
//    debugBF->addTerm((1/dt) * m2, vm2);
//    BasisCachePtr cell0Cache = BasisCache::basisCacheForCell(mesh, 0);
//    ElementTypePtr elemType = mesh->getElementType(0);
//    Intrepid::FieldContainer<double> stiffness(1,elemType->testOrderPtr->totalDofs(),elemType->trialOrderPtr->totalDofs());
//    Intrepid::FieldContainer<double> cellSideParities(1,2);
//    bool rowMajor = false;
//    bool checkForZeroCols = false;
//    debugBF->stiffnessMatrix(stiffness, elemType, cellSideParities, cell0Cache, rowMajor, checkForZeroCols);
//    cout << "stiffness matrix: \n"<< stiffness << endl;
//  }
  
  printConservationReport();
  double t = 0;
  int timeStepNumber = 0;
  
  // for the moment, time step adjustment only reduces time step size
  auto adjustTimeStep = [&]() -> void
  {
    if (spaceTime) return; // no time step to adjust...
    // Check that dt is reasonable vis-a-vis CFL
    double h = (x_b-x_a) / meshWidth / polyOrder;
    FunctionPtr soundSpeed = Function::sqrtFunction(gamma * p_soln / rho_soln);
    FunctionPtr fluidSpeed = velocityMagnitude_soln;
    double maxSoundSpeed = soundSpeed->linfinitynorm(mesh);
    double maxFluidSpeed = fluidSpeed->linfinitynorm(mesh);
    double dtCFL = h / (maxSoundSpeed + maxFluidSpeed);
    if (dt > dtCFL)
    {
      if (rank == 0)
      {
        cout << "Time step " << dt << " exceeds CFL-stable value of " << dtCFL << endl;
      }
      while (dt > dtCFL)
      {
        dt /= 2.0;
      }
      numTimeSteps = (finalTime - t) / dt + timeStepNumber;
      if (rank == 0)
      {
        cout << "Set time step to " << dt;
        cout << "; set numTimeSteps to " << numTimeSteps << endl;
      }
    }
    else
    {
      cout << "Time step " << dt << " is below CFL-stable value of " << dtCFL << endl;
    }
  };
  
  adjustTimeStep();
  
  for (timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    int maxNonlinearSteps = 100;
    double alpha = 1.0;
    while (((l2NormOfIncrement > nonlinearTolerance) || (alpha < 1.0)) && (stepNumber < maxNonlinearSteps))
    {
      alpha = form->solveAndAccumulate();
      int solveCode = form->getSolveCode();
      if (solveCode != 0)
      {
        if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
        exit(1);
      }
      l2NormOfIncrement = form->L2NormSolutionIncrement();
      if (rank == 0)
      {
        std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
        if (alpha != 1.0)
        {
          std::cout << " (alpha = " << alpha << ")" << std::endl;
        }
        else
        {
          std::cout << std::endl;
        }
      }
      
      stepNumber++;
      solutionHistoryExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
      solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
    }
    if (l2NormOfIncrement > nonlinearTolerance)
    {
      if (rank == 0)
      {
        cout << "Nonlinear iteration failed.  Exiting...\n";
      }
      return -1;
    }
    t += dt;
    
    printLocalConservationReport(); // since this depends on the difference between current/previous solution, we need to call before we set prev to current.
    solutionExporter.exportSolution(form->solution(),functionsToPlot,functionNames,t); // similarly, since the entropy compares current and previous, need this to happen before setSolution()
    printConservationReport();
    
    if (rank == 0) std::cout << "========== t = " << t << ", time step number " << timeStepNumber+1 << " ==========\n";
    if (timeStepNumber != numTimeSteps - 1)
    {
      form->solutionPreviousTimeStep()->setSolution(form->solution());
    }
  }
  
  // now that we're at final time, let's output pressure, velocity, density in a format suitable for plotting
  functionsToPlot.push_back(rho);
  functionNames.push_back("density");
  
  writeFunctions(mesh, meshWidth, polyOrder, x_a, functionsToPlot, functionNames, solnName.str());
  
  return 0;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  int meshWidth = 200;
  int polyOrder = 2;
  int delta_k   = 3;
  bool useCondensedSolve = false; // condensed solve UNSUPPORTED for now; before turning this on, make sure the various Solution objects are all set to use this in a compatible way...
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  double nonlinearTolerance    = 1e-2;
  bool useSpaceTime = false;
  int temporalPolyOrder =  1;
  int temporalMeshWidth = -1;  // if useSpaceTime gets set to true and temporalMeshWidth is left unset, we'll use meshWidth = (finalTime / dt / temporalPolyOrder)
  
  double x_a   = -0.5;
  double x_b   = 0.5;

  // h is about 1/400; max speed of sound is about 1.4; the speed during the solve gets up to about 1.8
  // call the max characteristic speed about 4.0.
  // the CFL condition then would suggest 1/1600 -- about 0.000625 -- as the maximum time step
  double dt    = 0.0005; // time step
  
  bool enforceConservationUsingLagrangeMultipliers = false;
  
  bool runSodInstead = false;
  
  std::map<string, TestNormChoice> normChoices = {
    {"steadyGraph", STEADY_GRAPH_NORM},
    {"transientGraph", TRANSIENT_GRAPH_NORM},
    {"experimental", EXPERIMENTAL_CONSERVATIVE_NORM}
  };
  
  std::string normChoiceString = "transientGraph";
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth);
  cmdp.setOption("dt", &dt);
  cmdp.setOption("deltaP", &delta_k);
  cmdp.setOption("nonlinearTol", &nonlinearTolerance);
  cmdp.setOption("enforceConservation", "dontEnforceConservation", &enforceConservationUsingLagrangeMultipliers);
  cmdp.setOption("runSodInstead", "runBrioWu", &runSodInstead);
  cmdp.setOption("spaceTime","backwardEuler", &useSpaceTime);
  cmdp.setOption("temporalPolyOrder", &temporalPolyOrder);
  cmdp.setOption("temporalMeshWidth", &temporalMeshWidth);
  
  cmdp.setOption("norm", &normChoiceString);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  TestNormChoice normChoice;
  if (normChoices.find(normChoiceString) != normChoices.end())
  {
    normChoice = normChoices[normChoiceString];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
  
  double gamma = runSodInstead ? 1.4 : 2.0;
  double finalTime = runSodInstead ? 0.20 : .08;
  
  Teuchos::RCP<IdealMHDFormulation> form;
  if (useSpaceTime)
  {
    cout << "*********************************************************************************************************** \n";
    cout << "****** SPACE-TIME NOTE: to date, we haven't had much luck with space-time for Brio-Wu or Euler/Sod. ******* \n";
    cout << "*********************************************************************************************************** \n";
    double t0 = 0.0;
    double t1 = finalTime;
    if (temporalMeshWidth == -1)
    {
      temporalMeshWidth = (finalTime / dt / temporalPolyOrder);
    }
    auto spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalMeshWidth);
    form = IdealMHDFormulation::spaceTimeFormulation(spaceDim, spaceTimeMeshTopo, polyOrder, temporalPolyOrder, delta_k, gamma);
  }
  else
  {
    form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k, gamma);
  }

  return runSolver(form, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                   nonlinearTolerance, normChoice, enforceConservationUsingLagrangeMultipliers, runSodInstead, useSpaceTime);
}
