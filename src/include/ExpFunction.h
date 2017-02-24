// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  ExpFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ExpFunction_h
#define Camellia_ExpFunction_h

#include "Function.h"
#include "SimpleFunction.h"
#include "TypeDefs.h"

namespace Camellia
{
  template<class Scalar>
  class Exp : public TFunction<Scalar>
  {
  private:
    TFunctionPtr<Scalar> _power;
  public:
    Exp(TFunctionPtr<Scalar> power);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    TFunctionPtr<Scalar> dx();
    TFunctionPtr<Scalar> dy();
    TFunctionPtr<Scalar> dz();
    std::string displayString();
  };
  
  // explicitly instantiate Exp<double> class:
  template class Exp<double>;
  
class Exp_x : public SimpleFunction<double>
{
public:
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_y : public SimpleFunction<double>
{
public:
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_z : public SimpleFunction<double>
{
public:
  double value(double x, double y, double z);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_ax : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ax(double a);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

class Exp_ay : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ay(double a);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  string displayString();
};

class Exp_at : public SimpleFunction<double>
{
  double _a;
public:
  Exp_at(double a);
  double value(double x, double t);
  double value(double x, double y, double t);
  double value(double x, double y, double z, double t);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  TFunctionPtr<double> dt();
  string displayString();
};
}
#endif
