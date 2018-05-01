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
  TFunctionPtr<Scalar> column(int spaceDim, TFunctionPtr<Scalar> f,  int colNumber)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(f->rank() < 2, std::invalid_argument, "f must rank >= 2");
    vector<TFunctionPtr<Scalar>> components;
    for (int d=1; d<=spaceDim; d++)
    {
      auto row = f->spatialComponent(d);
      components.push_back(row->spatialComponent(colNumber));
    }
    return TFunction<Scalar>::vectorize(components);
  }
  
  template<typename Scalar>
  TFunctionPtr<Scalar> contraction(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "f1 and f2 must have like rank");
    if (f1->rank() == 0)
    {
      return f1 * f2;
    }
    TFunctionPtr<Scalar> sum;
    // f1 and f2 have the same rank >= 1, so we can recursively contract in each component
    sum = TFunction<Scalar>::zero();
    for (int d=1; d<=spaceDim; d++)
    {
      sum = sum + contraction(spaceDim, f1->spatialComponent(d), f2->spatialComponent(d));
    }
    return sum;
  }
  
  template<typename Scalar>
  TFunctionPtr<Scalar> dot(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
  {
    bool bothRank1      = (f1->rank() == 1) && (f2->rank() == 1);
    bool f1Rank1f2Rank2 = (f1->rank() == 1) && (f2->rank() == 2);
    bool f1Rank2f2Rank1 = (f1->rank() == 2) && (f2->rank() == 1);
    bool valid = bothRank1 || f1Rank1f2Rank2 || f1Rank2f2Rank1;
    TEUCHOS_TEST_FOR_EXCEPTION(!valid, std::invalid_argument, "dot requires either that f1 and f2 both have rank 1, or one of them be rank 1 and the other rank 2");
    int rank = bothRank1 ? 0 : 1;
    auto sum = TFunction<Scalar>::zero(rank);
    for (int d=1; d<=spaceDim; d++)
    {
      sum = sum + f1->spatialComponent(d) * f2->spatialComponent(d);
    }
    return sum;
  }
  
  // ! returns the identity matrix in the provided number of spatial dimensions (a rank-2 function)
  template<typename Scalar>
  TFunctionPtr<Scalar> identityMatrix(int spaceDim)
  {
    vector<TFunctionPtr<Scalar>> rows;
    for (int d1=1; d1<=spaceDim; d1++)
    {
      vector<TFunctionPtr<Scalar>> rowEntries;
      for (int d2=1; d2<=spaceDim; d2++)
      {
        auto entry = (d1 == d2) ? TFunction<Scalar>::constant(1.0) : TFunction<Scalar>::zero();
        rowEntries.push_back(entry);
      }
      auto row = TFunction<Scalar>::vectorize(rowEntries);
      rows.push_back(row);
    }
    return TFunction<Scalar>::vectorize(rows);
  }
  
  template<typename Scalar>
  TFunctionPtr<Scalar> matvec(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION((f1->rank() == 2) && (f2->rank() == 1), std::invalid_argument, "matvec requires f2 be rank 2, f1 be rank 1");
    std::vector<TFunctionPtr<Scalar>> components;
    for (int d=1; d<=spaceDim; d++)
    {
      components.push_back(dot(spaceDim, f1->spatialComponent(d), f2));
    }
    return TFunction<Scalar>::vectorize(components);
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
  template TFunctionPtr<double> column<double>         (int spaceDim, TFunctionPtr<double> f,  int colNumber);
  template TFunctionPtr<double> contraction<double>    (int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  template TFunctionPtr<double> dot<double>            (int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  template TFunctionPtr<double> identityMatrix<double> (int spaceDim);
  template TFunctionPtr<double> matvec<double>         (int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  template TFunctionPtr<double> outerProduct<double>   (int spaceDim, TFunctionPtr<double> f1, TFunctionPtr<double> f2);
}
