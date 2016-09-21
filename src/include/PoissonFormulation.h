// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PoissonFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#ifndef Camellia_PoissonFormulation_h
#define Camellia_PoissonFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia
{
class PoissonFormulation
{
public:
  enum PoissonFormulationChoice
  {
    CONTINUOUS_GALERKIN,
    PRIMAL,
    ULTRAWEAK
  };
private:
  BFPtr _poissonBF;
  int _spaceDim;

  static const string S_U;
  static const string S_SIGMA;

  static const string S_U_HAT;
  static const string S_SIGMA_N_HAT;

  static const string S_V;
  static const string S_TAU;
public:
  PoissonFormulation(int spaceDim, bool useConformingTraces, PoissonFormulationChoice formulationChoice=ULTRAWEAK);

  BFPtr bf();
  
  RHSPtr rhs(FunctionPtr forcingFunction);

  // field variables:
  VarPtr u();
  VarPtr sigma();

  // traces:
  VarPtr sigma_n_hat();
  VarPtr u_hat();

  // test variables:
  VarPtr v();
  VarPtr tau();
};
}

#endif