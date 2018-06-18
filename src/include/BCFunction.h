#ifndef CAMELLIA_BC_FUNCTION
#define CAMELLIA_BC_FUNCTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER


#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class BCFunction : public TFunction<Scalar>
{
  Intrepid::FieldContainer<bool> _imposeHere;
  int _varID;
  BCPtr _bc;
  TFunctionPtr<Scalar> _spatiallyFilteredFunction;
public:
  BCFunction(BCPtr bc, int varID, TFunctionPtr<Scalar> spatiallyFilteredFunction, int rank);
  void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  bool imposeOnCell(int cellIndex);
  int varID();

  TFunctionPtr<Scalar> curl();
  TFunctionPtr<Scalar> div();
  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();

  std::vector< TFunctionPtr<Scalar> > memberFunctions();
  
  static Teuchos::RCP<BCFunction<Scalar>> bcFunction(BCPtr bc, int varID);
};
}

#endif
