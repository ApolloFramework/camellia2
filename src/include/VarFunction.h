// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  VarFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/28/2018.
//
//

#ifndef Camellia_VarFunction_h
#define Camellia_VarFunction_h

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class VarFunction : public TFunction<Scalar>
{
  VarPtr _var;
  bool _weightFluxesBySideParity;
public:
  VarFunction(VarPtr var);
  
  // ! This is an abstract function, not valid for direct evaluation: hence, calling values will throw an exception.
  void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  
  std::string displayString();
  bool boundaryValueOnly();
  
  //! evaluates, filling in _var values using soln
  TFunctionPtr<Scalar> evaluateAt(SolutionPtr soln);
  
  //! returns the LinearTerm corresponding to _var
  TLinearTermPtr<Scalar> jacobian(TSolutionPtr<Scalar> soln);

  //! returns true
  bool isAbstract();
};
}

#endif
