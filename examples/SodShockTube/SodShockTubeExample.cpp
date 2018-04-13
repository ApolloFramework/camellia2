//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "CompressibleNavierStokesFormulationRefactor.hpp"
#include "CompressibleNavierStokesConservationForm.hpp"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

// template logic on the formulation so that we can use either the conservation formulation or the primitive variable formulation
template<class Form>
FunctionPtr energy(Teuchos::RCP<Form> form);

template<class Form>
FunctionPtr momentum(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr momentumFlux(Teuchos::RCP<Form> form); // the thing that we take a divergence of in the strong form (field variables only)

template<class Form>
FunctionPtr pressure(Teuchos::RCP<Form> form);

template<class Form>
FunctionPtr velocity(Teuchos::RCP<Form> form);

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
FunctionPtr energy<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
{
  FunctionPtr E   = Function::solution(form->E(),   form->solutionPreviousTimeStep());
  return E;
}

template<>
FunctionPtr energy<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  double Cv = form->Cv();
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr u   = Function::solution(form->u(1),  form->solutionPreviousTimeStep());
  FunctionPtr T   = Function::solution(form->T(),   form->solutionPreviousTimeStep());
  
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

template<class Form>
int runSolver(Teuchos::RCP<Form> form, bool conservationVariables, double dt, int meshWidth, double x_a, double x_b,
              int polyOrder, int cubatureEnrichment, bool useCondensedSolve, double nonlinearTolerance, int continuationSteps, double continuationTolerance)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  const int spaceDim = 1;
  double Re = 1.0 / form->mu();
  
  int numTimeSteps = 0.20 / dt; // standard for Sod is to take it to t = 0.20
  
  if (rank == 0)
  {
    using namespace std;
    cout << "Solving with:\n";
    cout << "Re = " << Re << endl;
    cout << "p  = " << polyOrder << endl;
    cout << "dt = " << dt << endl;
    cout << meshWidth << " elements; " << numTimeSteps << " timesteps.\n";
  }
  
  form->setTimeStep(dt);
  form->solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
  form->solutionIncrement()->setCubatureEnrichmentDegree(cubatureEnrichment);
  
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
    initialState[form->q(1)->ID()]    = Function::zero();
    initialState[form->D(1,1)->ID()]  = Function::zero();
    
    // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
    // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
    initialState[form->T_hat()->ID()] = T;
    initialState[form->tc()->ID()]    = rho * u * n_x;
    initialState[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
    
    initialState[form->u_hat(1)->ID()] = u;
    initialState[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
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
  solnName << "sodShockSolutionRe" << Re << "_dt" << dt << "_k" << polyOrder;
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
  form->addVelocityTraceCondition(     leftX, Function::constant(u_a));
  form->addVelocityTraceCondition(    rightX, Function::constant(u_b));
  form->addTemperatureTraceCondition(  leftX, Function::constant(T_a));
  form->addTemperatureTraceCondition( rightX, Function::constant(T_b));
  
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr m   = momentum(form, true); // true: previous time step
  FunctionPtr E   = energy(form);
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
  
//  BFPtr bf = form->solutionIncrement()->bf();
//  LinearTermPtr residual = bf->testFunctional(form->solutionIncrement()) - form->solutionIncrement()->rhs()->linearTermCopy();
//  VarPtr vm = form->vm(1);
//  LinearTermPtr momentumResidualTerm = residual->getPartMatchingVariable(vm);
//  map<int, FunctionPtr> vmEqualsOneMap = {{vm->ID(), Function::constant(1.0)}};
//  FunctionPtr momentumResidualFunction = momentumResidualTerm->evaluate(vmEqualsOneMap);
//
//  auto printMomentumLocalConservationReport = [&]() -> void
//  {
//    auto & myCellIDs = mesh->cellIDsInPartition();
//    int myCellCount = myCellIDs.size();
//    Intrepid::FieldContainer<double> cellIntegrals(myCellCount);
//    int cellOrdinal = 0;
//    double maxConservationFailure = 0.0;
//    for (auto cellID : myCellIDs)
//    {
//      double cellIntegral = momentumResidualFunction->integrate(cellID, mesh);
//      cellIntegrals[cellOrdinal] = cellIntegral;
//      maxConservationFailure = std::max(abs(cellIntegral), maxConservationFailure);
//    }
//    cout << "On rank " << rank << ", max cellwise momentum conservation failure: " << maxConservationFailure << endl;
//  };
  // The above local momentum conservation computation may be correct; I'm not sure.
  // If you turn it on, the reported values are very high at times (implausibly so, is my thought).
  // Might do better to formulate by hand.
  /*
   Something like:
   */
  FunctionPtr rho_currentTime = Function::solution(form->rho(), form->solution());
  FunctionPtr momentum_soln = momentum(form, false); // false: not previous time step, but current
  FunctionPtr momentum_prev = momentum(form, true);
  FunctionPtr momentumTimeStep = (momentum_soln - momentum_prev) / dt;
  FunctionPtr momentumFlux = Function::solution(form->tm(1), form->solution(), true); // true: include sideParity weights
  auto printMomentumLocalConservationReport = [&]() -> void
  {
    auto & myCellIDs = mesh->cellIDsInPartition();
    int myCellCount = myCellIDs.size();
    Intrepid::FieldContainer<double> cellIntegrals(myCellCount);
    int cellOrdinal = 0;
    double maxConservationFailure = 0.0;
    for (auto cellID : myCellIDs)
    {
      double timeDifferenceIntegral = momentumTimeStep->integrate(cellID, mesh);
      double fluxIntegral = momentumFlux->integrate(cellID, mesh);
      double cellIntegral = timeDifferenceIntegral + fluxIntegral;
//      cout << "timeDifferenceIntegral = " << timeDifferenceIntegral << endl;
//      cout << "fluxIntegral           = " << fluxIntegral << endl;
      cellIntegrals[cellOrdinal] = cellIntegral;
      maxConservationFailure = std::max(abs(cellIntegral), maxConservationFailure);
      cellOrdinal++;
    }
    cout << "On rank " << rank << ", max cellwise momentum conservation failure: " << maxConservationFailure << endl;
//    for (int cellOrdinal=0; cellOrdinal<myCellCount; cellOrdinal++)
//    {
//      cout << "cell ordinal " << cellOrdinal << ", conservation failure: " << cellIntegrals[cellOrdinal] << endl;
//    }
  };
  
  
  printConservationReport();
  double t = 0;
  double Re_0 = std::min(1e0,Re); // for continuation in Reynolds number
  for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    int continuationStepNumber = 0;
    int maxNonlinearSteps = 10;
    form->setMu(1.0 / Re_0);
    if (rank == 0) cout << "for continuation, set Re to " << Re_0 << endl;
    double Re_current = Re_0;
    double Re_multiplier = pow(Re / Re_0, 1./continuationSteps);
    while (((continuationStepNumber < continuationSteps) || (l2NormOfIncrement > nonlinearTolerance)) && (stepNumber < maxNonlinearSteps))
    {
      double alpha = form->solveAndAccumulate();
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
    t += dt;
    
    printMomentumLocalConservationReport(); // since this depends on the difference between current/previous solution, we need to call before we set prev to current.
    form->solutionPreviousTimeStep()->setSolution(form->solution());
    solutionExporter.exportSolution(form->solutionPreviousTimeStep(),functionsToPlot,functionNames,t);
    
    printConservationReport();
    if (rank == 0) std::cout << "========== t = " << t << ", time step number " << timeStepNumber+1 << " ==========\n";
  }
  
  //  // create a refStrategy, just for the purpose of uniform refinement.
  //  // (The RHS is likely incorrect for refinement purposes, because of the treatment of fluxes.  We can fix this; see NavierStokesVGPFormulation for how...)
  //  double energyThreshold = 0.2;
  //  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(form->solutionIncrement(), energyThreshold);
  //
  //  int numUniformRefinements = 0;
  //  int stepOffset = stepNumber;
  //  for (int refNumber = 0; refNumber < numUniformRefinements; refNumber++)
  //  {
  //    std::cout << "**** Performing Uniform Refinement ****\n";
  //    refStrategy->hRefineUniformly();
  //    stepNumber = 0;
  //    l2NormOfIncrement = 1.0;
  //    while ((l2NormOfIncrement > nonlinearTolerance) && (stepNumber < maxNonlinearSteps))
  //    {
  //      double alpha = form->solveAndAccumulate();
  //      int solveCode = form->getSolveCode();
  //      if (solveCode != 0)
  //      {
  //        if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
  //        exit(1);
  //      }
  //      l2NormOfIncrement = form->L2NormSolutionIncrement();
  //      std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
  //      std::cout << " (alpha = " << alpha << ")" << std::endl;
  //
  //      stepNumber++;
  //      solutionExporter.exportSolution(form->solution(), double(stepNumber + stepOffset));  // use stepNumber as the "time" value for export...
  //      solutionIncrementExporter.exportSolution(form->solutionIncrement(), double(stepNumber + stepOffset));
  //    }
  //    stepOffset += stepNumber;
  //  }
  
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
  double dt    = 0.005; // time step
  
  bool useConservationFormulation = true;
  
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
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
  
  bool useConformingTraces = true;
  if (useConservationFormulation)
  {
    auto form = CompressibleNavierStokesConservationForm::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                  meshTopo, polyOrder, delta_k);
    return runSolver(form, useConservationFormulation, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                     nonlinearTolerance, continuationSteps, continuationTolerance);
  }
  else
  {
    auto form = CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                     meshTopo, polyOrder, delta_k);
    return runSolver(form, useConservationFormulation, dt, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                     nonlinearTolerance, continuationSteps, continuationTolerance);
  }
}
