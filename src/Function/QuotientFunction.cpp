//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "QuotientFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
QuotientFunction<Scalar>::QuotientFunction(TFunctionPtr<Scalar> f, TFunctionPtr<Scalar> scalarDivisor) : TFunction<Scalar>( f->rank() )
{
  if ( scalarDivisor->rank() != 0 )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
  if (scalarDivisor->isZero())
  {
    cout << "WARNING: division by zero in QuotientFunction.\n";
  }
}

template <typename Scalar>
bool QuotientFunction<Scalar>::boundaryValueOnly()
{
  return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
}

template <typename Scalar>
string QuotientFunction<Scalar>::displayString()
{
  ostringstream ss;
  // use memberFunctions as a proxy for whether we should include parentheses (for sums and products, this does what we want)
  bool includeParentheses = _scalarDivisor->memberFunctions().size() > 1;
  if (includeParentheses)
    ss << _f->displayString() << " / (" << _scalarDivisor->displayString() << ")";
  else
    ss << _f->displayString() << " / " << _scalarDivisor->displayString();
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::evaluateAt(SolutionPtr soln)
{
  auto f1 = Function::evaluateAt(_f, soln);
  auto f2 = Function::evaluateAt(_scalarDivisor, soln);
  return f1 / f2;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dx()
{
  if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dy()
{
  if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dz()
{
  if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dt()
{
  if ( (_f->dt().get() == NULL) || (_scalarDivisor->dt().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dt() / _scalarDivisor - _f * _scalarDivisor->dt() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TLinearTermPtr<Scalar> QuotientFunction<Scalar>::jacobian(TSolutionPtr<Scalar> soln)
{
  auto f1 = Function::evaluateAt(_f, soln);
  auto f2 = Function::evaluateAt(_scalarDivisor, soln);
  auto df1 = _f->jacobian(soln);
  auto df2 = _scalarDivisor->jacobian(soln);
  return (1.0 / f2) * df1 - (1.0 / ( f2 * f2 )) * df2;
}

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> QuotientFunction<Scalar>::memberFunctions()
{
  return {{_f, _scalarDivisor}};
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::t()
{
  return _f->t() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::x()
{
  return _f->x() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::y()
{
  return _f->y() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::z()
{
  return _f->z() / _scalarDivisor;
}

namespace Camellia
{
template class QuotientFunction<double>;
}
