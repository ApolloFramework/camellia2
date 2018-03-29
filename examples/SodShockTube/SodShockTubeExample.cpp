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
  int polyOrder = 2;
  int delta_k   = 1; // 1 is likely sufficient in 1D
  bool useCondensedSolve = true;
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  
  double x_a   = -0.5;
  double x_b   = 0.5;
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);

  double Re    = 1e3;   // Reynolds number
  double dt    = 0.005; // time step
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth);
  cmdp.setOption("Re", &Re);
  cmdp.setOption("dt", &dt);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
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
    initialState[form->T()->ID()]     = T;
    initialState[form->u(1)->ID()]    = u;
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
  
  // project the initial state both onto the solution object representing the previous time state,
  // as well as the current state (the latter is the initial guess for the current time step).
  const int solutionOrdinal = 0;
  form->solutionPreviousTimeStep()->projectOntoMesh(initialState, solutionOrdinal);
  form->solution()->projectOntoMesh(initialState, solutionOrdinal);
  
  // history export gets every nonlinear increment as a separate step
  HDF5Exporter solutionHistoryExporter(form->solutionIncrement()->mesh(), "sodShockSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(form->solutionIncrement()->mesh(), "sodShockSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  solnName << "sodShockSolutionRe" << Re << "_dt" << dt << "_k" << polyOrder;
  HDF5Exporter solutionExporter(form->solutionIncrement()->mesh(), solnName.str(), ".");
  
  solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), 0.0);
  solutionHistoryExporter.exportSolution(form->solution(), 0.0);
  solutionExporter.exportSolution(form->solutionPreviousTimeStep(), 0.0);
  
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

  // define the pressure so we can plot in our solution export
  
  FunctionPtr rho = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr T   = Function::solution(form->T(), form->solutionPreviousTimeStep());
  FunctionPtr p = R * rho * T;
  
  double t = 0;
  double Re_0 = 1.0; // for continuation in Reynolds number
  for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    double nonlinearTolerance = 1e-2;
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    int maxNonlinearSteps = 20;
    form->setMu(1.0 / Re_0);
    cout << "for continuation, set Re to " << Re_0 << endl;
    double Re_current = Re_0;
    int continuationSteps = 3;
    double Re_multiplier = pow(Re / Re_0, 1./continuationSteps);
    while (((stepNumber < continuationSteps) || (l2NormOfIncrement > nonlinearTolerance)) && (stepNumber < maxNonlinearSteps))
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
      solutionHistoryExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
      solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
      if (stepNumber <= continuationSteps)
      {
        Re_current *= Re_multiplier;
        form->setMu(1./Re_current);
        cout << "for continuation, set Re to " << Re_current << endl;
      }
    }
    t += dt;
    form->solutionPreviousTimeStep()->setSolution(form->solution());
    solutionExporter.exportSolution(form->solutionPreviousTimeStep(),{p},{"pressure"},t);
    
    std::cout << "========== t = " << t << ", time step number " << timeStepNumber << " ==========\n";
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
