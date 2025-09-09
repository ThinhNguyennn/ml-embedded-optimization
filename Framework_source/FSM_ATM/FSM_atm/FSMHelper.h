#ifndef FSMHELPER_H
#define FSMHELPER_H

class FsmHelper
{
public:
    /*
     * @brief function toStateName would return the name of the input state
     * @param[in] state input
     * @return state name
     */
    static const char *toStateName(int state);
};

#endif //FSMHELPER_H