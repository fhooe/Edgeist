/**
 * @file
 * @brief Implements the base object for all classes.
 */

#ifndef OBJECT_H
#define OBJECT_H

namespace Edgeist {
/**
 * @brief Base object for all classes.
 **/
class object {
public:
    virtual ~object() = default;

protected:
    object() = default;
};
} // namespace Edgeist

#endif // OBJECT_H
