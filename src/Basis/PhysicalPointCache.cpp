//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  PhysicalPointCache.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#include "PhysicalPointCache.h"

using namespace Intrepid;
using namespace Camellia;

PhysicalPointCache::PhysicalPointCache(const FieldContainer<double> &physCubPoints) : BasisCache()
{
  _physCubPoints = physCubPoints;
}
const FieldContainer<double> & PhysicalPointCache::getPhysicalCubaturePoints()   // overrides super
{
  return _physCubPoints;
}
int PhysicalPointCache::getSpaceDim()
{
  return _physCubPoints.dimension(2);
}
FieldContainer<double> & PhysicalPointCache::writablePhysicalCubaturePoints()   // allows overwriting the contents
{
  return _physCubPoints;
}