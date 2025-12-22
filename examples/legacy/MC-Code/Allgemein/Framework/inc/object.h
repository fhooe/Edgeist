#ifndef OBJECT_H
#define OBJECT_H

//common root class for all other classes.
//If there is no base class, this class is to be selected
class object
{
public:

	virtual ~object() = default;

protected:

	object() = default;

};

#endif // !OBJECT_H
