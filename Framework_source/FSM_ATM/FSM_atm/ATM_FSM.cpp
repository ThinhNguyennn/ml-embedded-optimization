#include "ATM_FSM.h"

using namespace fsmlite;


/**************************************************************************************
 * send_model_application(const char* message)
 * Function: Send page want to change to Application
 * state: 3 model state
 * Return: None
***************************************************************************************/
void send_model_application(const char* message) {
    //!< key 56789 is key id for page message between FSM and Application
    key_t key = 56789;

    //!< Creating or connect to Queue
    int msgid_sender = msgget(key, IPC_CREAT | IPC_EXCL | 0666);

    //!< If the queue already exists, do not use the IPC_CREAT flag and do not check for errors
    if (msgid_sender == -1 && errno == EEXIST) {
        msgid_sender = msgget(key, 0666);
    } else if (msgid_sender == -1) {    
        perror("msgget");  //!< if error connect, print the error
        return;
    }

    //!< Send page message to Application
    msg_buffer_send msg_send;
    msg_send.msg_type = 1;
    strncpy(msg_send.msg_text, message, sizeof(msg_send.msg_text) - 1);
    msg_send.msg_text[sizeof(msg_send.msg_text) - 1] = '\0';  //!< Ensure null-terminated

    //!< start sending
    if (msgsnd(msgid_sender, &msg_send, sizeof(msg_send.msg_text), 0) == -1) {
        perror("msgsnd");
        return;
    } else {
        std::cout << "send complete" << std::endl;
    }
}

/**************************************************************************************
* 
*                WORDSPACE FOR FUNCTION OF FINITE STATE MACHINE
* 
***************************************************************************************/

/**************************************************************************************
 * function name: CNN_Act()
 * Function: the main action in the CNN state
 * state: CNN state
 * Return: None
***************************************************************************************/
void UI_Model::CNN_Act()
{
    std::cout << "In model CNN" << std::endl;
    send_model_application("CNN");
}

/**************************************************************************************
 * function name: BNN_Act()
 * Function: the main action in the BNN state
 * state: BNN state
 * Return: None
***************************************************************************************/
void UI_Model::BNN_Act()
{
    std::cout << "In model BNN" << std::endl;
    send_model_application("BNN");
}

/**************************************************************************************
 * function name: LSVM_Act()
 * Function: the main action in the LSVM state
 * state: LSVM state
 * Return: None
***************************************************************************************/
void UI_Model::LSVM_Act()
{
    std::cout << "In model LSVM" << std::endl;
    send_model_application("LSVM");
}
