//
//  CompressibleNavierStokesProblem.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/15/18.
//

#include "CompressibleNavierStokesProblem.hpp"

#include "Function.h"
#include "TrigFunctions.h"
#include "PolarizedFunction.h"

using namespace Camellia;
using namespace std;

class Exp_ay2 : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ay2(double a) : _a(a) {}
  double value(double x, double y)
  {
    return exp(_a*y*y);
  }
};

class Log_ay2b : public SimpleFunction<double>
{
  double _a;
  double _b;
public:
  Log_ay2b(double a, double b) : _a(a), _b(b) {}
  double value(double x, double y)
  {
    return log(_a*y*y+_b);
  }
};

//class SqrtFunction : public Function {
//private:
//  FunctionPtr _function;
//public:
//  SqrtFunction(FunctionPtr function) : Function(0), _function(function) {}
//  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
//    int numCells = values.dimension(0);
//    int numPoints = values.dimension(1);
//
//    _function->values(values, basisCache);
//
//    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        values(cellIndex, ptIndex) = sqrt(values(cellIndex, ptIndex));
//      }
//    }
//  }
//};
//
//class BoundedSqrtFunction : public Function {
//private:
//  FunctionPtr _function;
//  double _bound;
//public:
//  BoundedSqrtFunction(FunctionPtr function, double bound) : Function(0), _function(function), _bound(bound) {}
//  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
//    int numCells = values.dimension(0);
//    int numPoints = values.dimension(1);
//
//    _function->values(values, basisCache);
//
//    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        values(cellIndex, ptIndex) = sqrt(std::max(values(cellIndex, ptIndex),_bound));
//      }
//    }
//  }
//};

FunctionPtr CompressibleNavierStokesProblem::rhoInitial()
{
  return Function::constant(1.0);
}

FunctionPtr CompressibleNavierStokesProblem::TInitial()
{
  return Function::constant(1.0);
}

vector<FunctionPtr> CompressibleNavierStokesProblem::uInitial()
{
  return vector<FunctionPtr>(3,Function::zero());
}


class NohProblem : public CompressibleNavierStokesProblem
{
public:
  // use default (unity) initial values for rho, T
  vector<FunctionPtr> uInitial()
  {
    FunctionPtr cos_y = Teuchos::rcp(new Cos_ay(1.0));
    FunctionPtr sin_y = Teuchos::rcp(new Sin_ay(1.0));
    FunctionPtr cos_theta = Teuchos::rcp( new PolarizedFunction<double>( cos_y ) );
    FunctionPtr sin_theta = Teuchos::rcp( new PolarizedFunction<double>( sin_y ) );
    FunctionPtr zero = Function::zero();
    
    return {-cos_theta, -sin_theta, zero};
  }
};

class RayleighTaylorProblem : public CompressibleNavierStokesProblem
{
  FunctionPtr pInitial()
  {
    // pressure, I assume
    double g = -1;
    double beta = 20;
    double pi = atan(1)*4;
    double rho1 = 1;
    double rho2 = 2;
    FunctionPtr atan_betay = Teuchos::rcp(new ArcTan_ay(beta));
    
    FunctionPtr y = Function::yn(1);
    double C = 4. + (1.5+1./pi*atan(beta)-1./(2*pi*beta)*log(beta*beta+1));
    FunctionPtr log_b2y21 = Teuchos::rcp(new Log_ay2b(beta*beta,1));
    FunctionPtr pInitial = g*((rho1+rho2)/2.*y + (rho2-rho1)/pi*(atan_betay*y-1./(2*beta)*log_b2y21))+C;
    return pInitial;
  }
public:
  FunctionPtr rhoInitial()
  {
    double beta = 20;
    double pi = atan(1)*4;
    double rho1 = 1;
    double rho2 = 2;
    FunctionPtr atan_betay = Teuchos::rcp(new ArcTan_ay(beta));
    
    return (rho1+rho2)/2. + (rho2-rho1)/pi*atan_betay;
  }
  FunctionPtr TInitial()
  {
    return 1./.4 * pInitial()/ rhoInitial();
  }
  
  vector<FunctionPtr> uInitial()
  {
    double u0 = 0.02;
    double pi = atan(1.)*4.;
    FunctionPtr cos_2pix = Teuchos::rcp(new Cos_ax(2*pi));
    FunctionPtr sin_2pix = Teuchos::rcp(new Sin_ax(2*pi));
    FunctionPtr exp_m2piy2 = Teuchos::rcp(new Exp_ay2(-2*pi));
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1_init = u0*exp_m2piy2*2*y*sin_2pix;
    FunctionPtr u2_init = u0*exp_m2piy2*2*y*cos_2pix;
  }
};

class TriplePointProblem : public CompressibleNavierStokesProblem
{
public:
  FunctionPtr rhoInitial()
  {
    return 1.0 - (1-0.125)*Function::heaviside(1)*Function::heavisideY(1.5);
  }
  FunctionPtr TInitial()
  {
    return 1.0 - (1-0.1)*Function::heaviside(1);
  }
  // use default of 0.0 as initial value for u
};

Teuchos::RCP<CompressibleNavierStokesProblem> CompressibleNavierStokesProblem::noh()
{
  return Teuchos::rcp( new NohProblem() );
}

Teuchos::RCP<CompressibleNavierStokesProblem> CompressibleNavierStokesProblem::rayleighTaylor()
{
  return Teuchos::rcp( new RayleighTaylorProblem() );
}
Teuchos::RCP<CompressibleNavierStokesProblem> CompressibleNavierStokesProblem::triplePoint()
{
  return Teuchos::rcp( new TriplePointProblem() );
}

Teuchos::RCP<CompressibleNavierStokesProblem> CompressibleNavierStokesProblem::namedProblem(const string &problemName)
{
  if (problemName == "Noh")
  {
    return noh();
  }
  else if (problemName == "RayleighTaylor")
  {
    return rayleighTaylor();
  }
  else if (problemName == "TriplePoint")
  {
    return triplePoint();
  }
  else
  {
    std::cout << "Unknown problem name: " << problemName << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown problem name");
  }
}

