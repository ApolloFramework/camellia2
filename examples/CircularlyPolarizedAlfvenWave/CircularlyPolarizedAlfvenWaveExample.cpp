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
#include "TrigFunctions.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

const double PI  = 3.141592653589793238462;

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
              bool spaceTime)
{
  MeshPtr mesh = form->solutionIncrement()->mesh();
  int spaceDim = mesh->getDimension();
  
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
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  else if (normChoice == EXPERIMENTAL_CONSERVATIVE_NORM)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  
  double finalTime = 1.0;
  int numTimeSteps = spaceTime ? 1 : finalTime / dt;
  
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
  }
  
  form->setTimeStep(dt);
  solnIncrement->setUseCondensedSolve(useCondensedSolve);
  solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichment);
//  solnIncrement->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
//  solnIncrement->setWriteRHSToMatrixMarketFile(   true, "/tmp/b.dat");
  
  double gamma = form->gamma();
  double c_v   = form->Cv();
  
  // set up initial conditions for 1D problem
  auto rhoInitial       = Function::constant(1.0);
  auto BParallel        = Function::constant(1.0);
  auto BPerpendicular   = Function::constant(0.1);
  auto direction        = Function::constant(1.0);
  auto pressure         = Function::zero();
  auto vParallel        = Function::zero();
  
  double x1size, x2size, x3size;
  double sin_a2, cos_a2;
  double sin_a3, cos_a3;
  double ang_2, ang_3;
  if (spaceDim == 1)
  {
    x1size = 1.0;
    x2size = 0.0;
    x3size = 0.0;
    ang_2  = 0.0;
    ang_3  = 0.0;
    
    sin_a2 = sin(ang_2);
    cos_a2 = cos(ang_2);
    
    sin_a3 = sin(ang_3);
    cos_a3 = cos(ang_3);
  }
  else if (spaceDim == 2)
  {
    x1size = 2.2360680;
    x2size = 1.1180399;
    x3size = 1.0;
    ang_3 = atan(x1size/x2size);
    sin_a3 = sin(ang_3);
    cos_a3 = cos(ang_3);
    
    ang_2 = atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
    
    sin_a2 = 0.0;
    cos_a2 = 1.0;
  }
  else // (spaceDim == 3)
  {
    x1size = 3.0;
    x2size = 1.5;
    x3size = 1.5;
    ang_3 = atan(x1size/x2size);
    sin_a3 = sin(ang_3);
    cos_a3 = cos(ang_3);
    
    ang_2 = atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
    sin_a2 = sin(ang_2);
    cos_a2 = cos(ang_2);
  }
  
  double x1 = x1size*cos_a2*cos_a3;
  double x2 = x2size*cos_a2*sin_a3;
//  double x3 = x3size*sin_a2; // x3 is unused in any dimension...
  double lambda = x1;
  if ((spaceDim > 1) && (x2 < x1)) {
    lambda = x2;
  }
  double k_par = 2.0*PI/lambda;
  
  FunctionPtr v_A            = BParallel      / Function::sqrtFunction(rhoInitial);
  FunctionPtr vPerpendicular = BPerpendicular / Function::sqrtFunction(rhoInitial);
  FunctionPtr fac            = direction;
  
  FunctionPtr x,y,z;
  x = Function::xn(1);
  if (spaceDim == 1)
  {
    y = Function::zero();
    z = Function::zero();
  }
  else if (spaceDim == 2)
  {
    y = Function::yn(1);
    z = Function::zero();
  }
  else if (spaceDim == 3)
  {
    y = Function::yn(1);
    z = Function::zn(1);
  }
  
  auto theta = cos_a2*(x*cos_a3 + y*sin_a3) + z*sin_a2;
  auto sn = TrigFunctions<double>::sin(k_par*theta);
  auto cs = fac*TrigFunctions<double>::cos(k_par*theta);
  
  auto Mx = rhoInitial*vParallel;
  auto My = -fac*rhoInitial*vPerpendicular*sn;
  auto Mz = -rhoInitial*vPerpendicular*cs;
  
  auto m1Initial = Mx*cos_a2*cos_a3 - My*sin_a3 - Mz*sin_a2*cos_a3;
  auto m2Initial = Mx*cos_a2*sin_a3 + My*cos_a3 - Mz*sin_a2*sin_a3;
  auto m3Initial = Mx*sin_a2                    + Mz*cos_a2;
  auto vInitial = Function::vectorize(m1Initial / rhoInitial, m2Initial / rhoInitial, m3Initial / rhoInitial);
  
  auto bx = BParallel;
  auto by = BPerpendicular*sn;
  auto bz = BPerpendicular*cs;
  
  FunctionPtr BxInitial, ByInitial, BzInitial;
  if (spaceDim == 1)
  {
    BxInitial = bx*cos_a2*cos_a3 + by*sin_a3 + bz*sin_a2*cos_a3; // signs for 2nd, 3rd terms reversed from 1D input deck -- trust 2D input deck for these
    ByInitial = bx*cos_a2*sin_a3 - by*cos_a3 + bz*sin_a2*sin_a3; // signs for 2nd, 3rd terms reversed from 1D input deck -- trust 2D input deck for these
    BzInitial = bx*sin_a2                    - bz*cos_a2;        // signs for last term reversed from 1D input deck -- trust 2D input deck for these
  }
  else
  {
    BxInitial = bx*cos_a2*cos_a3 + by*sin_a3 + bz*sin_a2*cos_a3; // signs for 2nd, 3rd terms reversed from 1D input deck -- trust 2D input deck for these
    ByInitial = bx*cos_a2*sin_a3 - by*cos_a3 + bz*sin_a2*sin_a3; // signs for 2nd, 3rd terms reversed from 1D input deck -- trust 2D input deck for these
    BzInitial = bx*sin_a2                    - bz*cos_a2;        // signs for last term reversed from 1D input deck -- trust 2D input deck for these
  }
  auto BInitial  = Function::vectorize(BxInitial, ByInitial, BzInitial);
  
  auto EInitial = pressure / (gamma - 1.) + 0.5 * dot(3,vInitial,vInitial) + 0.5 * dot(3,BInitial,BInitial);
  
  double R  = form->R();
  double Cv = form->Cv();
  
  if (rank == 0)
  {
    cout << "R =     " << R << endl;
    cout << "Cv =    " << Cv << endl;
    cout << "gamma = " << gamma << endl;
    cout << "Initial State:\n";
    cout << "rho = " << rhoInitial->displayString() << endl;
    cout << "E   = " << EInitial->displayString()   << endl;
    cout << "Bx  = " << BxInitial->displayString()  << endl;
    cout << "By  = " << ByInitial->displayString()  << endl;
    cout << "Bz  = " << BzInitial->displayString()  << endl;
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  {
    // for Ideal MHD, even in 1D, u is a vector, as is B
    form->setInitialCondition(rhoInitial, vInitial, EInitial, BInitial);
//    if (spaceTime)
//    {
//      // have had some issues using the discontinuous initial guess for all time in Sod problem at least
//      // let's try something much more modest: just unit values for rho, E, B, zero for u
//      FunctionPtr one    = Function::constant(1.0);
//      FunctionPtr rhoOne = one;
//      FunctionPtr EOne   = one;
//      FunctionPtr BGuess;
//      if (!runSodInstead)
//      {
//        BGuess   = Function::vectorize(Bx, one, one);
//      }
//      else
//      {
//        BGuess   = Function::vectorize(zero, zero, zero);
//      }
//      auto initialGuess = form->exactSolutionFieldMap(rhoOne, velocityVector, EOne, BGuess);
//      form->setInitialState(initialGuess);
//    }
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
  HDF5Exporter solutionHistoryExporter(mesh, "cpAlfvenSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(mesh, "cpAlfvenSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  solnName << "cpAlfvenSolution" << "_dt" << dt << "_k" << polyOrder;
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

  auto velocityVector = Function::vectorize(Function::zero(), Function::zero(), Function::zero());
  
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
  
  if (spaceDim == 1)
  {
    writeFunctions(mesh, meshWidth, polyOrder, x_a, functionsToPlot, functionNames, solnName.str());
  }
  
  // output error in By:
  FunctionPtr ByFinal = Function::solution(form->B(2), form->solution());
  FunctionPtr err_By  = ByFinal - ByInitial;
  double err_By_L2 = err_By->l2norm(mesh);
  double err_By_L1 = err_By->l1norm(mesh);
  
  if (rank == 0)
  {
    cout << "Final L^1 error of By: " << err_By_L1 << endl;
    cout << "Final L^2 error of By: " << err_By_L2 << endl;
  }
  
  return 0;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  int meshWidth = 32;
  int polyOrder = 2;
  int delta_k   = 1;
  bool useCondensedSolve = false; // condensed solve UNSUPPORTED for now; before turning this on, make sure the various Solution objects are all set to use this in a compatible way...
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  double nonlinearTolerance    = 1e-2;
  bool useSpaceTime = false;
  int temporalPolyOrder =  1;
  int temporalMeshWidth = -1;  // if useSpaceTime gets set to true and temporalMeshWidth is left unset, we'll use meshWidth = (finalTime / dt / temporalPolyOrder)
  
  double x_a   = -0.5;
  double x_b   = 0.5;

  double dt    = 0.004; // time step
  
  bool enforceConservationUsingLagrangeMultipliers = false;
  
  std::map<string, TestNormChoice> normChoices = {
    {"steadyGraph", STEADY_GRAPH_NORM},
    {"transientGraph", TRANSIENT_GRAPH_NORM},
    {"experimental", EXPERIMENTAL_CONSERVATIVE_NORM}
  };
  
  std::string normChoiceString = "transientGraph";
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth); // x mesh width; y width will be 1/2 this in 2D
  cmdp.setOption("spaceDim",  &spaceDim);
  cmdp.setOption("dt", &dt);
  cmdp.setOption("deltaP", &delta_k);
  cmdp.setOption("nonlinearTol", &nonlinearTolerance);
  cmdp.setOption("enforceConservation", "dontEnforceConservation", &enforceConservationUsingLagrangeMultipliers);
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
  
  bool periodicBCs = true;
  MeshTopologyPtr meshTopo;
  if (spaceDim == 1)
  {
    meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth, periodicBCs);
  }
  else if (spaceDim == 2)
  {
    double width  = 2.2360680;
    double height = 1.1180399;
    double x0     = 0.0;
    double y0     = 0.0;
    int horizontalElements = meshWidth;
    int verticalElements   = meshWidth / 2;
    bool divideIntoTriangles = false;
    vector<PeriodicBCPtr> periodicBCs;
    periodicBCs.push_back(PeriodicBC::xIdentification(x0, width));
    periodicBCs.push_back(PeriodicBC::yIdentification(y0, height));
    meshTopo = MeshFactory::quadMeshTopology(width, height, horizontalElements, verticalElements, divideIntoTriangles,
                                             x0, y0, periodicBCs);
  }
  else if (spaceDim == 3)
  {
    double width  = 3.0;
    double height = 1.5;
    double depth  = 1.5;
    vector<double> meshDims = {width,height,depth};
    double x0     = 0.0;
    double y0     = 0.0;
    double z0     = 0.0;
    vector<double> meshOrigin = {x0,y0,z0};
    int horizontalElements = meshWidth;
    int verticalElements   = meshWidth / 2;
    int depthElements      = meshWidth / 2;
    vector<int> elementCounts = {horizontalElements, verticalElements, depthElements};
    vector<PeriodicBCPtr> periodicBCs;
    periodicBCs.push_back(PeriodicBC::xIdentification(x0, width));
    periodicBCs.push_back(PeriodicBC::yIdentification(y0, height));
    periodicBCs.push_back(PeriodicBC::zIdentification(z0, depth));
    meshTopo = MeshFactory::rectilinearMeshTopology(meshDims, elementCounts, meshOrigin, periodicBCs);
  }
  
  double gamma = 1.6667;
  double finalTime = 1.0;
  
  Teuchos::RCP<IdealMHDFormulation> form;
  if (useSpaceTime)
  {
    cout << "****************************************************************************************** \n";
    cout << "****** SPACE-TIME NOTE: to date, we haven't tried space-time for CP Alfven waves. ******** \n";
    cout << "****************************************************************************************** \n";
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
                   nonlinearTolerance, normChoice, enforceConservationUsingLagrangeMultipliers, useSpaceTime);
}
