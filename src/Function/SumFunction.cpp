//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
#include "SumFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
SumFunction<Scalar>::SumFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>(f1->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

template <typename Scalar>
bool SumFunction<Scalar>::boundaryValueOnly()
{
  // if either summand is BVO, then so is the sum...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

template <typename Scalar>
string SumFunction<Scalar>::displayString()
{
  ostringstream ss;
  ss << "(" << _f1->displayString() << " + " << _f2->displayString() << ")";
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::evaluateAt(SolutionPtr soln)
{
  auto f1 = Function::evaluateAt(_f1, soln);
  auto f2 = Function::evaluateAt(_f2, soln);
  return f1 + f2;
}

template <typename Scalar>
TLinearTermPtr<Scalar> SumFunction<Scalar>::jacobian(TSolutionPtr<Scalar> soln)
{
  auto df1 = _f1->jacobian(soln);
  auto df2 = _f2->jacobian(soln);
  return df1 + df2;
}

template <typename Scalar>
void SumFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  _f1->values(values,basisCache);
  _f2->addToValues(values,basisCache);
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::x()
{
  if ( (_f1->x() == Teuchos::null) || (_f2->x() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->x() + _f2->x();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::y()
{
  if ( (_f1->y() == Teuchos::null) || (_f2->y() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->y() + _f2->y();
}
template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::z()
{
  if ( (_f1->z() == Teuchos::null) || (_f2->z() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->z() + _f2->z();
}
template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::t()
{
  if ( (_f1->t() == Teuchos::null) || (_f2->t() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->t() + _f2->t();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dx()
{
  if ( (_f1->dx() == Teuchos::null) || (_f2->dx() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->dx() + _f2->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dy()
{
  if ( (_f1->dy() == Teuchos::null) || (_f2->dy() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dy() + _f2->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dz()
{
  if ( (_f1->dz() == Teuchos::null) || (_f2->dz() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dz() + _f2->dz();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dt()
{
  if ( (_f1->dt() == Teuchos::null) || (_f2->dt() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dt() + _f2->dt();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::grad(int numComponents)
{
  if ( (_f1->grad(numComponents) == Teuchos::null) || (_f2->grad(numComponents) == Teuchos::null) )
  {
    return Teuchos::null;
  }
  else
  {
    return _f1->grad(numComponents) + _f2->grad(numComponents);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::div()
{
  if ( (_f1->div() == Teuchos::null) || (_f2->div() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  else
  {
    return _f1->div() + _f2->div();
  }
}

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> SumFunction<Scalar>::memberFunctions()
{
  return {{_f1, _f2}};
}

namespace Camellia
{
template class SumFunction<double>;
}

