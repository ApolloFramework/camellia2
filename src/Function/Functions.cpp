//
//  Functions.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/28/18.
//

#include "Functions.hpp"

#include "Function.h"

namespace Camellia
{
  template<typename Scalar>
  TFunctionPtr<Scalar> contraction(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "f1 and f2 must have like rank");
    if (f1->rank() == 0)
    {
      return f1 * f2;
    }
    // f1 and f2 have the same rank >= 1, so we can recursively contract in each component
    TFunctionPtr<Scalar> sum = TFunction<Scalar>::zero();
    for (int d=1; d<=spaceDim; d++)
    {
      sum = sum + contraction(spaceDim, f1->spatialComponent(d), f2->spatialComponent(d));
    }
    return sum;
  }
  
  template<typename Scalar>
  TFunctionPtr<Scalar> outerProduct(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != 1, std::invalid_argument, "f1 must have rank 1");
    TEUCHOS_TEST_FOR_EXCEPTION(f2->rank() != 1, std::invalid_argument, "f2 must have rank 1");
    // vectors are understood as column vectors; outer product is f1 * (f2)^T
    // consequence of this is that when we ask for f->x(), we refer to a *row*.
    // the rank-2 container will thus have as its x component the row corresponding to f1->x() * f2
    vector<TFunctionPtr<Scalar>> rows;
    for (int d=1; d<=spaceDim; d++)
    {
      rows.push_back(f1->spatialComponent(d) * f2);
    }
    return TFunction<Scalar>::vectorize(rows);
  }
}

// ETIs below
namespace Camellia {
  template TFunctionPtr<double> contraction<double> (int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  template TFunctionPtr<double> outerProduct<double>(int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
}
