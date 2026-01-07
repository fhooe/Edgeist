/**
 * @file
 * @brief Implements the base object for all classes
 */

#ifndef OBJECT_H
#define OBJECT_H

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

#endif // OBJECT_H
