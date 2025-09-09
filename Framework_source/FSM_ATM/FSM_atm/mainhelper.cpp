#include "mainhelper.h"


/*------------------------------- FUNCTION CHANGE PAGE ------------------------------*/

/**************************************************************************************
  * function name: data_change_page()
  * Function: Get infor_page from the app interface
  * state: 4 states of interface
  * Return: std::string
***************************************************************************************/
std::string data_change_model() {
    //!< key 12345 is key id for btn message between FSM and Application
    key_t key = 12345;

    //!< Create a message queue or connect to an existing message queue
    int msgid_receiver = msgget(key, IPC_CREAT | 0666);
    if (msgid_receiver == -1) {
        perror("msgget");
        return "Error getting message queue";
    }

    //!< Receive the message
    msg_buffer_receiver msg;
    if (msgrcv(msgid_receiver, &msg, sizeof(msg.msg_text), 1, 0) == -1) {
        perror("msgrcv");
    }
    
    std::string result(msg.msg_text, sizeof(msg.msg_text));
    size_t nullPos = result.find_first_of('\0');
    if(nullPos != std::string::npos)
    {
      result.erase(nullPos);
    }
    return result;
}

/**************************************************************************************
  * function name: Process_model_Changeloop(std::string current_page)
  * Function: processing changes page
  * state: PROCESSING state
  * Return: None
***************************************************************************************/
void Process_model_Changeloop(std::string current_page)
{
  UI_Model UI_Model;
  //!< Changing page when start page = "CNN"
  if ( (current_page == "BNN") && (UI_Model.GetCurrentState() == "CNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_CNNtoBNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "LSVM") && (UI_Model.GetCurrentState() == "CNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_CNNtoLSVM_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "CNN") && (UI_Model.GetCurrentState() == "CNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_CNNtoCNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  }

  //!< Changing page when start page = "BNN"
  else if ( (current_page == "CNN") && (UI_Model.GetCurrentState() == "BNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_BNNtoCNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "LSVM") && (UI_Model.GetCurrentState() == "BNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_BNNtoLSVM_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "BNN") && (UI_Model.GetCurrentState() == "BNN") )
  {
    UI_Model.process_event(UI_Model::Change_model_BNNtoBNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  }
  
  //!< Changing page when start page = "LSVM"
  else if ( (current_page == "CNN") && (UI_Model.GetCurrentState() == "LSVM") )
  {
    UI_Model.process_event(UI_Model::Change_model_LSVMtoCNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "BNN") && (UI_Model.GetCurrentState() == "LSVM") )
  {
    UI_Model.process_event(UI_Model::Change_model_LSVMtoBNN_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } else if ( (current_page == "LSVM") && (UI_Model.GetCurrentState() == "LSVM") )
  {
    UI_Model.process_event(UI_Model::Change_model_LSVMtoLSVM_Evt());
    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;
  } 
}
