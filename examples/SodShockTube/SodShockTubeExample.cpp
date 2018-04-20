//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "CompressibleNavierStokesFormulationRefactor.hpp"
#include "CompressibleNavierStokesConservationForm.hpp"
#include "LagrangeConstraints.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

// template logic on the formulation so that we can use either the conservation formulation or the primitive variable formulation
template<class Form>
FunctionPtr energy(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr momentum(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr momentumFlux(Teuchos::RCP<Form> form); // the thing that we take a divergence of in the strong form (field variables only)

template<class Form>
FunctionPtr pressure(Teuchos::RCP<Form> form);

template<class Form>
FunctionPtr velocity(Teuchos::RCP<Form> form);

template<class Form>
void addConservationConstraint(Teuchos::RCP<Form> form);

template<>
FunctionPtr pressure<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
{
  // pressure = (gamma - 1) * (E - 1/2 * m dot m / rho)
  double gamma = form->gamma();
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr m1  = Function::solution(form->m(1),  form->solutionPreviousTimeStep());
  FunctionPtr E   = Function::solution(form->E(),   form->solutionPreviousTimeStep());
  
  FunctionPtr p = (gamma - 1) * (E - 0.5 * m1 * m1 / rho);
  return p;
}

template<>
FunctionPtr pressure<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  double R = form->R();
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr T   = Function::solution(form->T(), form->solutionPreviousTimeStep());
  FunctionPtr p = R * rho * T;
  return p;
}

template<>
FunctionPtr energy<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr E   = Function::solution(form->E(), soln);
  return E;
}

template<>
FunctionPtr energy<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  double Cv = form->Cv();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr u   = Function::solution(form->u(1),  soln);
  FunctionPtr T   = Function::solution(form->T(),   soln);
  
  FunctionPtr E = rho * (Cv * T + 0.5 * u * u);
  return E;
}

template<>
FunctionPtr momentum<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr m   = Function::solution(form->m(1), soln);
  return m;
}

template<>
FunctionPtr momentum<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr u   = Function::solution(form->u(1),  soln);
  
  FunctionPtr m = rho * u;
  return m;
}

template<>
FunctionPtr momentumFlux<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
{
  FunctionPtr rho = Function::solution(form->rho(),  form->solutionPreviousTimeStep());
  FunctionPtr m   = Function::solution(form->m(1),   form->solutionPreviousTimeStep());
  FunctionPtr E   = Function::solution(form->E(),    form->solutionPreviousTimeStep());
  FunctionPtr D   = Function::solution(form->D(1,1), form->solutionPreviousTimeStep());
  
  FunctionPtr p = pressure(form);
  FunctionPtr sigma = (D + D - 2./3. * D);
  FunctionPtr mFlux = m * m / rho + p - sigma;
  return mFlux;
}

template<>
FunctionPtr momentumFlux<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr u   = Function::solution(form->u(1),  form->solutionPreviousTimeStep());
  FunctionPtr D   = Function::solution(form->D(1,1), form->solutionPreviousTimeStep());
  
  FunctionPtr p   = pressure(form);
  
  FunctionPtr sigma = (D + D - 2./3. * D);
  
  FunctionPtr mFlux = rho * u * u + p - sigma;
  return mFlux;
}

template<>
FunctionPtr velocity<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
{
  FunctionPtr rho = Function::solution(form->rho(),  form->solutionPreviousTimeStep());
  FunctionPtr m   = Function::solution(form->m(1),   form->solutionPreviousTimeStep());
  return m / rho;
}

template<>
FunctionPtr velocity<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  FunctionPtr u   = Function::solution(form->u(1),   form->solutionPreviousTimeStep());
  return u;
}

template<>
void addConservationConstraint(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  cout << "WARNING: addConservationConstraint unimplemented for primitive-variable formulation...\n";
  // TODO: implement this
}

template<>
void addConservationConstraint(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
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

template<class Form>
void setVelocityOrMomentum(Teuchos::RCP<Form> form, std::map<int, FunctionPtr> &initialStateMap, FunctionPtr u, FunctionPtr rho, FunctionPtr T);

template<class Form>
void setTempOrEnergy(Teuchos::RCP<Form> form, std::map<int, FunctionPtr> &initialStateMap, FunctionPtr u, FunctionPtr rho, FunctionPtr T);

template<>
void setVelocityOrMomentum<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                     FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->m(1)->ID()]    = u * rho;
}

template<>
void setVelocityOrMomentum<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                        FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->u(1)->ID()]    = u;
}

template<>
void setTempOrEnergy<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, std::map<int, FunctionPtr> &initialStateMap,
                                                               FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  double Cv = form->Cv();
  FunctionPtr E = rho * (Cv * T + 0.5 * u * u);
  initialStateMap[form->E()->ID()] = E;
}

template<>
void setTempOrEnergy<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                        FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->T()->ID()]     = T;
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
  GRAPH_NORM,
  EXPERIMENTAL_CONSERVATIVE_NORM
};

template<class Form>
int runSolver(Teuchos::RCP<Form> form, bool conservationVariables, double dt, int meshWidth, double x_a, double x_b,
              int polyOrder, int cubatureEnrichment, bool useCondensedSolve, double nonlinearTolerance, int continuationSteps,
              double continuationTolerance, TestNormChoice normChoice, bool enforceConservationUsingLagrangeMultipliers)
{
  if (enforceConservationUsingLagrangeMultipliers)
  {
    addConservationConstraint(form);
  }
  int rank = Teuchos::GlobalMPISession::getRank();
  const int spaceDim = 1;
  double mu = form->mu();
  bool pureEuler = (mu == 0.0);
  
  SolutionPtr solnIncrement = form->solutionIncrement();
  SolutionPtr soln = form->solution();
    
  double Re = (!pureEuler) ? 1.0 / form->mu() : -1.0 ;
  
  if (normChoice == EXPERIMENTAL_CONSERVATIVE_NORM)
  {
    IPPtr ip = Teuchos::rcp(new IP);
    VarPtr vc  = form->vc();
    VarPtr vm  = form->vm(1);
    VarPtr ve  = form->ve();
    
    ip->addBoundaryTerm(vc);
    ip->addTerm(vc->dx()); // would be grad for spaceDim > 1
    
    ip->addBoundaryTerm(vm);
    ip->addTerm(vm->dx()); // would be grad for spaceDim > 1
    
    ip->addBoundaryTerm(ve);
    ip->addTerm(ve->dx()); // would be grad for spaceDim > 1
    
    if (!pureEuler)
    {
      // naive norm for the rest
      VarPtr S   = form->S(1);
      VarPtr tau = form->tau();
      
      ip->addTerm(S->dx());
      ip->addTerm(S);
      
      ip->addTerm(tau);
      ip->addTerm(tau->dx());
    }
    
    form->solutionIncrement()->setIP(ip);
  }
  
  int numTimeSteps = 0.20 / dt; // standard for Sod is to take it to t = 0.20
  
  if (rank == 0)
  {
    using namespace std;
    cout << "Solving with:\n";
    if (pureEuler)
      cout << "pure Euler\n";
    else
      cout << "Re = " << Re << endl;
    cout << "p  = " << polyOrder << endl;
    cout << "dt = " << dt << endl;
    cout << meshWidth << " elements; " << numTimeSteps << " timesteps.\n";
  }
  
  form->setTimeStep(dt);
  solnIncrement->setUseCondensedSolve(useCondensedSolve);
  solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichment);
//  solnIncrement->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
//  solnIncrement->setWriteRHSToMatrixMarketFile(   true, "/tmp/b.dat");
  
  double gamma = form->gamma();
  double c_v   = form->Cv();
  
  double rho_a = 1.0; // prescribed density at left
  double u_a   = 0.0; // Mach number
  double p_a   = 1.0; // prescribed pressure at left
  double T_a   = p_a / (rho_a * (gamma - 1.) * c_v);
  
  double rho_b = 0.125;
  double p_b   = 0.1;
  double u_b   = 0.0;
  double T_b   = p_b / (rho_b * (gamma - 1.) * c_v);
  
  double R  = form->R();
  double Cv = form->Cv();
  
  if (rank == 0)
  {
    cout << "R =   " << R << endl;
    cout << "Cv =  " << Cv << endl;
    cout << "State on left:\n";
    cout << "rho = " << rho_a << endl;
    cout << "p   = " << p_a   << endl;
    cout << "T   = " << T_a   << endl;
    cout << "u   = " << u_a   << endl;
    cout << "State on right:\n";
    cout << "rho = " << rho_b << endl;
    cout << "p   = " << p_b   << endl;
    cout << "T   = " << T_b   << endl;
    cout << "u   = " << u_b   << endl;
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  map<int, FunctionPtr> initialState;
  
  {
    auto H_right = Function::heaviside((x_a + x_b)/2.0); // Heaviside is 0 left of center, 1 right of center
    auto H_left  = 1.0 - H_right;  // this guy is 1 left of center, 0 right of center
    auto step = [&](double val_a, double val_b)
    {
      return H_left * val_a + H_right * val_b;
    };
    
    FunctionPtr rho = step(rho_a,rho_b);
    FunctionPtr T   = step(T_a, T_b);
    FunctionPtr u   = step(u_a, u_b);
    
    initialState[form->rho()->ID()]   = rho;
    setVelocityOrMomentum(form, initialState, u, rho, T);
    setTempOrEnergy      (form, initialState, u, rho, T);
    if (!pureEuler)
    {
      initialState[form->q(1)->ID()]    = Function::zero();
      initialState[form->D(1,1)->ID()]  = Function::zero();
    }
    
    // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
    // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
    initialState[form->tc()->ID()]    = rho * u * n_x;
    initialState[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
    initialState[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
    
    if (!pureEuler)
    {
      initialState[form->T_hat()->ID()] = T;
      initialState[form->u_hat(1)->ID()] = u;
    }
  }
  
  // define the pressure so we can plot in our solution export
  FunctionPtr p = pressure(form);
  // define u so we can plot in solution export (in case we're in conservation variables)
  FunctionPtr u = velocity(form);
  
  // project the initial state both onto the solution object representing the previous time state,
  // as well as the current state (the latter is the initial guess for the current time step).
  const int solutionOrdinal = 0;
  form->solutionPreviousTimeStep()->projectOntoMesh(initialState, solutionOrdinal);
  form->solution()->projectOntoMesh(initialState, solutionOrdinal);
  
  
  MeshPtr mesh = form->solutionIncrement()->mesh();
  
  vector<FunctionPtr> functionsToPlot;
  vector<string> functionNames;

  if (conservationVariables)
  {
    functionsToPlot = {p,u};
    functionNames   = {"pressure","velocity"};
  }
  else
  {
    functionsToPlot = {p};
    functionNames   = {"pressure"};
  }
  
  // history export gets every nonlinear increment as a separate step
  HDF5Exporter solutionHistoryExporter(mesh, "sodShockSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(mesh, "sodShockSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  if (!pureEuler)
    solnName << "sodShockSolutionRe" << Re << "_dt" << dt << "_k" << polyOrder;
  else
    solnName << "sodShockSolutionEuler" << "_dt" << dt << "_k" << polyOrder;
  if (conservationVariables) solnName << "_conservationVariables";
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
  
  // Borrowing some from Truman's dissertation code
  SpatialFilterPtr leftX  = SpatialFilter::matchingX(x_a);
  SpatialFilterPtr rightX = SpatialFilter::matchingX(x_b);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  
  if (rank == 0)
  {
    std::cout << "T_a = " << T_a << std::endl;
    std::cout << "T_b = " << T_b << std::endl;
  }
  //  (SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
  form->addMassFluxCondition             ( SpatialFilter::allSpace(), Function::zero());
  form->addEnergyFluxCondition           ( SpatialFilter::allSpace(), Function::zero());
  form->addMomentumFluxCondition(      leftX, Function::constant(rho_a), Function::constant(u_a), Function::constant(T_a));
  form->addMomentumFluxCondition(     rightX, Function::constant(rho_b), Function::constant(u_b), Function::constant(T_b));
  if (!pureEuler)
  {
    form->addVelocityTraceCondition(     leftX, Function::constant(u_a));
    form->addVelocityTraceCondition(    rightX, Function::constant(u_b));
    form->addTemperatureTraceCondition(  leftX, Function::constant(T_a));
    form->addTemperatureTraceCondition( rightX, Function::constant(T_b));
  }
  
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr m   = momentum(form, true); // true: previous time step
  FunctionPtr E   = energy(form, true);
  FunctionPtr tc  = Function::solution(form->tc(),  form->solutionPreviousTimeStep(), true);
  FunctionPtr tm  = Function::solution(form->tm(1), form->solutionPreviousTimeStep(), true);
  FunctionPtr te  = Function::solution(form->te(),  form->solutionPreviousTimeStep(), true);
  
  auto printConservationReport = [&]() -> void
  {
    double totalMass = rho->integrate(mesh);
    double totalMomentum = m->integrate(mesh);
    double totalEnergy = E->integrate(mesh);
    
//    double tc_left  = tc->evaluate(mesh, x_a);
//    double tc_right = tc->evaluate(mesh, x_b);
//    double tm_left  = tm->evaluate(mesh, x_a);
//    double tm_right = tm->evaluate(mesh, x_b);
//    double te_left  = te->evaluate(mesh, x_a);
//    double te_right = te->evaluate(mesh, x_b);
    if (rank == 0)
    {
      cout << "Total Mass:     " << totalMass << endl;
      cout << "Total Momentum: " << totalMomentum << endl;
      cout << "Total Energy:   " << totalEnergy << endl;
//      cout << "tc at left, right: " << tc_left << "," << tc_right << endl;
//      cout << "tm at left, right: " << tm_left << "," << tm_right << endl;
//      cout << "te at left, right: " << te_left << "," << te_right << endl;
    }
  };
  
  // elementwise local momentum conservation:
  FunctionPtr rho_soln = Function::solution(form->rho(), form->solution());
  FunctionPtr rho_prev = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr rhoTimeStep = (rho_soln - rho_prev) / dt;
  FunctionPtr rhoFlux = Function::solution(form->tc(), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr momentum_soln = momentum(form, false); // false: not previous time step, but current
  FunctionPtr momentum_prev = momentum(form, true);
  FunctionPtr momentumTimeStep = (momentum_soln - momentum_prev) / dt;
  FunctionPtr momentumFlux = Function::solution(form->tm(1), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr energy_soln = energy(form, false); // not previous time step, but current
  FunctionPtr energy_prev = energy(form, true);
  FunctionPtr energyTimeStep = (energy_soln - energy_prev) / dt;
  FunctionPtr energyFlux = Function::solution(form->te(), form->solution(), true); // include sideParity weights

//  vector<FunctionPtr> conservationFluxes = {momentumFlux};
//  vector<FunctionPtr> timeDifferences    = {momentumTimeStep};
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
  
  
  printConservationReport();
  double t = 0;
  double Re_0;
  for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    int continuationStepNumber = 0;
    int maxNonlinearSteps = 10;
    double Re_current, Re_multiplier;
    if (!pureEuler && (continuationSteps > 0))
    {
      if (timeStepNumber == 0)
      {
        Re_0 = std::min(1e2,Re); // for continuation in Reynolds number
      }
      else
      {
        Re_0 = std::min(1e4,Re); // for continuation in Reynolds number
      }
      form->setMu(1.0 / Re_0);
      if (rank == 0) cout << "for continuation, set Re to " << Re_0 << endl;
      Re_current = Re_0;
      Re_multiplier = pow(Re / Re_0, 1./continuationSteps);
    }
    else
    {
      // if Euler, no continuation
      continuationStepNumber = continuationSteps;
    }
    double alpha = 1.0;
    while (((continuationStepNumber < continuationSteps) || (l2NormOfIncrement > nonlinearTolerance)) && (stepNumber < maxNonlinearSteps))
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
        std::cout << " (alpha = " << alpha << ")" << std::endl;
      }
      
      stepNumber++;
      solutionHistoryExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
      solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
      if ((continuationStepNumber < continuationSteps) && ((l2NormOfIncrement < continuationTolerance) || stepNumber == maxNonlinearSteps))
      {
        Re_current *= Re_multiplier;
        form->setMu(1./Re_current);
        if (rank == 0) cout << "for continuation, set Re to " << Re_current << endl;
        continuationStepNumber++;
        // since we have changed the Re number, reset the step counter, as well as the l2NormOfIncrement
        l2NormOfIncrement = 1.0;
        stepNumber = 0;
      }
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
    form->solutionPreviousTimeStep()->setSolution(form->solution());
    solutionExporter.exportSolution(form->solutionPreviousTimeStep(),functionsToPlot,functionNames,t);
    
    printConservationReport();
    if (rank == 0) std::cout << "========== t = " << t << ", time step number " << timeStepNumber+1 << " ==========\n";
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
  int delta_k   = 1; // 1 is likely sufficient in 1D
  bool useCondensedSolve = false; // condensed solve UNSUPPORTED for now; before turning this on, make sure the various Solution objects are all set to use this in a compatible way...
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  double nonlinearTolerance    = 1e-2;
  double continuationTolerance = 1e-2;
  int continuationSteps = 4;
  
  double x_a   = -0.5;
  double x_b   = 0.5;

  double Re    = 1e8;   // Reynolds number
  double dt    = 0.001; // time step
  
  bool useConservationFormulation = true;
  bool useEuler = false;
  bool useExperimentalConservationNorm = false;
  bool runWriteFunctionTest = false; // option for debugging some output code...
  bool enforceConservationUsingLagrangeMultipliers = false;
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth);
  cmdp.setOption("Re", &Re);
  cmdp.setOption("dt", &dt);
  cmdp.setOption("deltaP", &delta_k);
  cmdp.setOption("continuationSteps", &continuationSteps);
  cmdp.setOption("nonlinearTol", &nonlinearTolerance);
  cmdp.setOption("continuationTol", &continuationTolerance);
  cmdp.setOption("conservationVariables", "primitiveVariables", &useConservationFormulation);
  cmdp.setOption("euler", "fullNS", &useEuler);
  cmdp.setOption("enforceConservation", "dontEnforceConservation", &enforceConservationUsingLagrangeMultipliers);
  
  cmdp.setOption("experimentalNorm", "graphNorm", &useExperimentalConservationNorm);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  TestNormChoice normChoice;
  if (useExperimentalConservationNorm)
  {
    normChoice = EXPERIMENTAL_CONSERVATIVE_NORM;
  }
  else
  {
    normChoice = GRAPH_NORM;
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
  
  bool useConformingTraces = true;
  
  if (runWriteFunctionTest)
  {
    auto form = CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                       meshTopo, polyOrder, delta_k);
    vector<FunctionPtr> functions = {Function::xn(1)};
    vector<string> functionNames  = {"linear"};
    writeFunctions(form->solution()->mesh(), meshWidth, polyOrder, -0.5, functions, functionNames, "testWrite");
    cout << "wrote test plot data; exiting...\n";
    return 0;
  }
  
  if (useConservationFormulation)
  {
    if (useEuler)
    {
      auto form = CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                         meshTopo, polyOrder, delta_k);
      return runSolver(form, useConservationFormulation, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
    else
    {
      auto form = CompressibleNavierStokesConservationForm::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                    meshTopo, polyOrder, delta_k);
      return runSolver(form, useConservationFormulation, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
  }
  else
  {
    auto form = CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                     meshTopo, polyOrder, delta_k);
    return runSolver(form, useConservationFormulation, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                     nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
  }
}
