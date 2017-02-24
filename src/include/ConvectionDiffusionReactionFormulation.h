// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  ConvectionDiffusionReactionFormulation.h
//  Camellia
//
//  Created by Nate Roberts on Mar 3, 2016.
//
//

#ifndef Camellia_ConvectionDiffusionReactionFormulation_h
#define Camellia_ConvectionDiffusionReactionFormulation_h

#include "BF.h"
#include "ParameterFunction.h"
#include "TypeDefs.h"
#include "VarFactory.h"

namespace Camellia
{
  class ConvectionDiffusionReactionFormulation
  {
  public:
    enum FormulationChoice
    {
      PRIMAL,
      SUPG,
      ULTRAWEAK,
      LEAST_SQUARES
    };
  private:
    FormulationChoice _formulationChoice;
    
    int _spaceDim;
    double _alpha; // _alpha: weight on the reaction term
    FunctionPtr _epsilon; // diffusion parameter
    FunctionPtr _beta;
    
    Teuchos::RCP<ParameterFunction> _stabilizationWeight; // for SUPG
    
    VarFactoryPtr _vf;
    BFPtr _bf;
    
    static const string S_U;
    static const string S_SIGMA;
    
    static const string S_UHAT;
    static const string S_SIGMA_N;
    
    static const string S_V;
    static const string S_TAU;
    
    void init(FormulationChoice formulation, int spaceDim,
              FunctionPtr beta, FunctionPtr epsilon, FunctionPtr sqrt_epsilon, double alpha);
  public:
    ConvectionDiffusionReactionFormulation(FormulationChoice formulation, int spaceDim,
                                           FunctionPtr beta, double epsilon, double alpha);
    
    ConvectionDiffusionReactionFormulation(FormulationChoice formulation, int spaceDim,
                                           FunctionPtr beta, FunctionPtr epsilon,
                                           FunctionPtr sqrt_epsilon, double alpha);
    
    VarFactoryPtr vf();
    BFPtr bf();
    
    // field variables:
    VarPtr u();
    VarPtr sigma();
    
    // traces:
    VarPtr sigma_n();
    VarPtr u_hat();
    
    // test variables:
    VarPtr v();
    VarPtr tau();
    
    // ! returns the stabilization weight used in the SUPG formulation (will be null for non-SUPG formulations).
    FunctionPtr SUPGStabilizationWeight();
    
    // ! sets the stabilization weight for SUPG formulations
    void setSUPGStabilizationWeight(FunctionPtr stabilizationWeight);
    
    // ! Returns the forcing function corresponding to the specified exact solution u
    FunctionPtr forcingFunction(FunctionPtr u_exact);
    
    // ! Returns the residual corresponding to the provided solution
    LinearTermPtr residual(SolutionPtr soln);
    
    // ! Returns the RHS corresponding to the provided forcing function
    RHSPtr rhs(FunctionPtr forcingFunction);
    
    // ! Returns either u or u_hat, whichever is appropriate for imposing Dirichlet BCs.
    VarPtr u_dirichlet();
  };
}

#endif
