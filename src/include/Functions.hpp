//
//  Functions.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/28/18.
//

#ifndef Functions_hpp
#define Functions_hpp

#include "TypeDefs.h"

namespace Camellia
{
  // ! Returns the componentwise contraction of f1 and f2, which must be of like rank
  template<typename Scalar>
  TFunctionPtr<Scalar> contraction(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  
  // ! Returns f1->x() * f2->x() + f1->y() * f2->y() …
  // ! Requires either that f1 and f2 both have rank 1, or one of them be rank 1 and the other rank 2.
  template<typename Scalar>
  TFunctionPtr<Scalar> dot(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  
  // ! returns the identity matrix in the provided number of spatial dimensions (a rank-2 function)
  template<typename Scalar>
  TFunctionPtr<Scalar> identityMatrix(int spaceDim);
  
  // ! f1 a rank-2 Function (a matrix, with x(), y(), z() as rows if spaceDim == 3)
  // ! f2 a rank-1 Function (a vector)
  // ! return the usual matrix-vector product (a vector)
  template<typename Scalar>
  TFunctionPtr<Scalar> matvec(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  
  // ! requires that f be rank >= 2.  colNumber is 1-based, and must be <= spaceDim.
  template<typename Scalar>
  TFunctionPtr<Scalar> column(int spaceDim, TFunctionPtr<Scalar> f, int colNumber);
  
  // ! requires that f1 and f2 be rank 1; returns their rank 2 outer product
  template<typename Scalar>
  TFunctionPtr<Scalar> outerProduct(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
}

#endif /* Functions_hpp */
