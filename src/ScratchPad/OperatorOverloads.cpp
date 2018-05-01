//
//  OperatorOverloads.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/30/18.
//

#include "OperatorOverloads.hpp"

#include "VarFunction.h"

using namespace Camellia;

namespace Camellia
{
  TFunctionPtr<double> operator*(VarPtr v1, VarPtr v2)
  {
    using Scalar = double;
    TFunctionPtr<Scalar> f1 = VarFunction<Scalar>::abstractFunction(v1);
    TFunctionPtr<Scalar> f2 = VarFunction<Scalar>::abstractFunction(v2);
    return f1 * f2;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator*(TLinearTermPtr<Scalar> lt,  VarPtr v)
  {
    auto f_lt = VarFunction<Scalar>::abstractFunction(lt);
    auto f_v  = VarFunction<Scalar>::abstractFunction(v);
    return f_lt * f_v;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator*(VarPtr v, TLinearTermPtr<Scalar> lt)
  {
    auto f_v  = VarFunction<Scalar>::abstractFunction(v);
    auto f_lt = VarFunction<Scalar>::abstractFunction(lt);
    return f_v * f_lt;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator*(TLinearTermPtr<Scalar> lt1, TLinearTermPtr<Scalar> lt2)
  {
    auto f_lt1 = VarFunction<Scalar>::abstractFunction(lt1);
    auto f_lt2 = VarFunction<Scalar>::abstractFunction(lt2);
    return f_lt1 * f_lt2;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator/(TLinearTermPtr<Scalar> lt,  VarPtr v)
  {
    auto f_lt = VarFunction<Scalar>::abstractFunction(lt);
    auto f_v  = VarFunction<Scalar>::abstractFunction(v);
    return f_lt / f_v;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator/(VarPtr v, TLinearTermPtr<Scalar> lt)
  {
    auto f_v  = VarFunction<Scalar>::abstractFunction(v);
    auto f_lt = VarFunction<Scalar>::abstractFunction(lt);
    return f_v / f_lt;
  }

  template<typename Scalar> TFunctionPtr<Scalar> operator/(TLinearTermPtr<Scalar> lt1, TLinearTermPtr<Scalar> lt2)
  {
    auto f_lt1 = VarFunction<Scalar>::abstractFunction(lt1);
    auto f_lt2 = VarFunction<Scalar>::abstractFunction(lt2);
    return f_lt1 / f_lt2;
  }
} // namespace Camellia


// ETIs below
namespace Camellia {
  template TFunctionPtr<double> operator*(TLinearTermPtr<double> lt,  VarPtr v);
  template TFunctionPtr<double> operator*(VarPtr v,                   TLinearTermPtr<double> lt);
  template TFunctionPtr<double> operator*(TLinearTermPtr<double> lt1, TLinearTermPtr<double> lt2);
  template TFunctionPtr<double> operator/(TLinearTermPtr<double> lt,  VarPtr v);
  template TFunctionPtr<double> operator/(VarPtr v,                   TLinearTermPtr<double> lt);
  template TFunctionPtr<double> operator/(TLinearTermPtr<double> lt1, TLinearTermPtr<double> lt2);
}
