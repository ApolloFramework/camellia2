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
  // ! requires that f1 and f2 be of like rank; contracts componentwise
  template<typename Scalar>
  TFunctionPtr<Scalar> contraction(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  
  // ! requires that f1 and f2 be rank 1; returns their rank 2 outer product
  template<typename Scalar>
  TFunctionPtr<Scalar> outerProduct(int spaceDim, TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
}

#endif /* Functions_hpp */
