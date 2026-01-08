/**
 * @file
 * @brief Implements the base object for all classes
 */

#ifndef EDGEIST_OBJECT_H
#define EDGEIST_OBJECT_H

namespace Edgeist {
/**
 * @brief Base object for all classes
 **/
class Object {
public:
    virtual ~Object() = default;

protected:
    Object() = default;
};
} // namespace Edgeist

#endif // EDGEIST_OBJECT_H
