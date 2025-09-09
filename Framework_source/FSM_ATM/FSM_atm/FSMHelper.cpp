#include "FSMHelper.h"
#include <iostream>

const char* FsmHelper::toStateName(int state)
{
    //todo: [TaiTN] clean up duplication
    typedef enum {
        CNN,                        
        BNN,                        
        LSVM       
    }state_t;
    switch (static_cast<state_t>(state))
    {
    case state_t::CNN:
        return "CNN";
    case state_t::BNN:
        return "BNN";
    case state_t::LSVM:
        return "LSVM";
        
    default:
        std::cout << "ERROR UNKNOWN STATE !" << std::endl;
        return "UNKNOWN";
    }
}