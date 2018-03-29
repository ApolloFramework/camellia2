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
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  int meshWidth = 400;
  int polyOrder = 1;
  int delta_k   = 1; // 1 is likely sufficient in 1D
  bool useCondensedSolve = true;
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  
  double x_a   = 0.0;
  double x_b   = 1.0;
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);

  double Re    = 1e2;   // Reynolds number
  double dt    = 0.01; // time step
  
  bool useConformingTraces = true;
  auto form = CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                   meshTopo, polyOrder, delta_k);
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
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  map<int, FunctionPtr> initialState;
  
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
  initialState[form->T()->ID()]     = T;
  initialState[form->u(1)->ID()]    = u;
  initialState[form->q(1)->ID()]    = Function::zero();
  initialState[form->D(1,1)->ID()]  = Function::zero();
  
  // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
  // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
  double R  = form->R();
  double Cv = form->Cv();
  initialState[form->T_hat()->ID()] = T;
  initialState[form->tc()->ID()]    = rho * u * n_x;
  initialState[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
  
  initialState[form->u_hat(1)->ID()] = u;
  initialState[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
  
  // project the initial state both onto the solution object representing the previous time state,
  // as well as the current state (the latter is the initial guess for the current time step).
  const int solutionOrdinal = 0;
  form->solutionPreviousTimeStep()->projectOntoMesh(initialState, solutionOrdinal);
  form->solution()->projectOntoMesh(initialState, solutionOrdinal);
  
  HDF5Exporter solutionExporter(form->solutionIncrement()->mesh(), "sodShockSteadySolution", ".");
  HDF5Exporter solutionIncrementExporter(form->solutionIncrement()->mesh(), "sodShockSteadySolutionIncrement", ".");
  
  double t = 0.0; // we're steady state; we're using t = 0.0 to indicate the Newton Step
  solutionIncrementExporter.exportSolution(form->solutionIncrement(), t); // set
  solutionExporter.exportSolution(form->solution(), t);
  
  // Borrowing some from Truman's dissertation code
  SpatialFilterPtr leftX  = SpatialFilter::matchingX(x_a);
  SpatialFilterPtr rightX = SpatialFilter::matchingX(x_b);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  
  std::cout << "T_a = " << T_a << std::endl;
  std::cout << "T_b = " << T_b << std::endl;
//  (SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
  form->addMassFluxCondition(         leftX, Function::constant(rho_a), Function::constant(u_a), Function::constant(T_a));
  form->addVelocityTraceCondition(    leftX, Function::constant(u_a)         );
  form->addTemperatureTraceCondition( leftX, Function::constant(T_a)         );
  form->addMassFluxCondition(        rightX, Function::constant(rho_b), Function::constant(u_b), Function::constant(T_b));
  form->addVelocityTraceCondition(   rightX, Function::constant(u_b)         );
  form->addTemperatureTraceCondition(rightX, Function::constant(T_b)         );
  
  VarPtr u1_hat = form->u_hat(1);

  double nonlinearTolerance = 1e-2;
  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;
  int maxNonlinearSteps = 20;
  while ((l2NormOfIncrement > nonlinearTolerance) && (stepNumber < maxNonlinearSteps))
  {
    double alpha = form->solveAndAccumulate();
    int solveCode = form->getSolveCode();
    if (solveCode != 0)
    {
      if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
      exit(1);
    }
    l2NormOfIncrement = form->L2NormSolutionIncrement();
    std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
    std::cout << " (alpha = " << alpha << ")" << std::endl;
    
    stepNumber++;
    solutionExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
    solutionIncrementExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
  }
  
  // create a refStrategy, just for the purpose of uniform refinement.
  // (The RHS is likely incorrect for refinement purposes, because of the treatment of fluxes.  We can fix this; see NavierStokesVGPFormulation for how...)
  double energyThreshold = 0.2;
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(form->solutionIncrement(), energyThreshold);
  
  int numUniformRefinements = 0;
  int stepOffset = stepNumber;
  for (int refNumber = 0; refNumber < numUniformRefinements; refNumber++)
  {
    std::cout << "**** Performing Uniform Refinement ****\n";
    refStrategy->hRefineUniformly();
    stepNumber = 0;
    l2NormOfIncrement = 1.0;
    while ((l2NormOfIncrement > nonlinearTolerance) && (stepNumber < maxNonlinearSteps))
    {
      double alpha = form->solveAndAccumulate();
      int solveCode = form->getSolveCode();
      if (solveCode != 0)
      {
        if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
        exit(1);
      }
      l2NormOfIncrement = form->L2NormSolutionIncrement();
      std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
      std::cout << " (alpha = " << alpha << ")" << std::endl;
      
      stepNumber++;
      solutionExporter.exportSolution(form->solution(), double(stepNumber + stepOffset));  // use stepNumber as the "time" value for export...
      solutionIncrementExporter.exportSolution(form->solutionIncrement(), double(stepNumber + stepOffset));
    }
    stepOffset += stepNumber;
  }
  
  return 0;
}
