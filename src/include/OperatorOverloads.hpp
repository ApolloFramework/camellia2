//
//  OperatorOverloads.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/30/18.
//

#ifndef OperatorOverloads_hpp
#define OperatorOverloads_hpp

#include "TypeDefs.h"

namespace Camellia
{
  // ********* Nonlinear Operator Overloads for Var/LinearTerm ********* //
  // (these return abstract Functions)

  // ! Since Var is *not* templated on Scalar type, we only support var * var for double types.
  // ! If you want Var * Var for things that are not double, you can use
  // !   Scalar one = 1.0;
  // !   auto product = (1.0 * v1) * v2;
  // ! or something similar
  TFunctionPtr<double> operator*(VarPtr v1, VarPtr v2);
  template<typename Scalar> TFunctionPtr<Scalar> operator*(TLinearTermPtr<Scalar> lt,  VarPtr v);
  template<typename Scalar> TFunctionPtr<Scalar> operator*(VarPtr v,                   TLinearTermPtr<Scalar> lt);
  template<typename Scalar> TFunctionPtr<Scalar> operator*(TLinearTermPtr<Scalar> lt1, TLinearTermPtr<Scalar> lt2);
  template<typename Scalar> TFunctionPtr<Scalar> operator/(TLinearTermPtr<Scalar> lt,  VarPtr v);
  template<typename Scalar> TFunctionPtr<Scalar> operator/(VarPtr v,                   TLinearTermPtr<Scalar> lt);
  template<typename Scalar> TFunctionPtr<Scalar> operator/(TLinearTermPtr<Scalar> lt1, TLinearTermPtr<Scalar> lt2);
} // namespace Camellia
#endif /* OperatorOverloads_hpp */
