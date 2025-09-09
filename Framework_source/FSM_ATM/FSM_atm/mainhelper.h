#ifndef MAINHELPER_H
#define MAINHELPER_H

#include "ATM_FSM.h"
#include "fsm.h"
#include "FSMHelper.h"
#include <iostream>
#include <sys/msg.h>
#include <cstring>

/**************************************************************************************
  * 
  * Wordspace: To set a MsgQ attribute
  *
***************************************************************************************/
struct msg_buffer_receiver {
    long msg_type;
    char msg_text[4]; 
};

/*----------------------------- DEFINE FUNCTION -----------------------------------*/   
std::string data_change_model();

void Process_model_Changeloop(std::string current_page);

#endif //MAINHELPER_H