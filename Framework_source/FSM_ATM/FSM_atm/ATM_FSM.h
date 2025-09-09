#include "fsm.h"
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <sys/ipc.h>
#include <sys/msg.h>

/**************************************************************************************
* 
*                WORDSPACE FOR DATATYPE OF FINITE STATE MACHINE
* 
***************************************************************************************/

    /*!
     * @brief type of message queue SEND 
     *        set the struct of message buffer sending
     */
    struct msg_buffer_send {
        long msg_type;
        char msg_text[5];
    };
    
class UI_Model: public fsmlite::fsm<UI_Model> {
    //! grant base class access to private transition_table
    friend class fsmlite::fsm<UI_Model>;

public:

    /*!
     * @brief States of the FSM in the project ATM Recycle
     *        The enum datatype contains the states of the FSM
     */
      typedef enum {
        CNN,                         
        BNN,                        
        LSVM       
      }states_t;

    UI_Model(state_type init_state = CNN) : fsm(init_state) {}

    /*!
     * @brief Simple event declare as below
     *        This contains only simple struct as an event of main Finite State Machine
     */
    //! @brief 
    struct CNN_Evt {};
    //! @brief 
    struct BNN_Evt {};
    //! @brief 
    struct LSVM_Evt {};
    //! @brief 


    /*!
     * @brief Simple event declare as below
     *         This contains only simple struct as an event of SubStateMachine [PROCESSING]
     */

    struct Change_model_CNNtoCNN_Evt {};

    struct Change_model_BNNtoBNN_Evt {};

    struct Change_model_LSVMtoLSVM_Evt {};

    struct Change_model_CNNtoBNN_Evt {};

    struct Change_model_CNNtoLSVM_Evt {};

    struct Change_model_BNNtoCNN_Evt {};

    struct Change_model_BNNtoLSVM_Evt {};

    struct Change_model_LSVMtoCNN_Evt {};

    struct Change_model_LSVMtoBNN_Evt {};

public:  

/**************************************************************************************
* 
*                WORDSPACE FOR FUNCTION OF FINITE STATE MACHINE
* 
***************************************************************************************
    
/**************************************************************************************
 * function name: CNN_Act()
 * Function: the main action in the CNN state
 * state: CNN state
 * Return: None
***************************************************************************************/
    void CNN_Act();

/**************************************************************************************
 * function name: BNN_Act()
 * Function: the main action in the BNN state
 * state: BNN state
 * Return: None
***************************************************************************************/
    void BNN_Act();

/**************************************************************************************
 * function name: LSVM_Act()
 * Function: the main action in the LSVM state
 * state: LSVM state
 * Return: None
***************************************************************************************/
    void LSVM_Act();

private:
    using m = UI_Model;  // for brevity
    
    using transition_table = table<
        //                 Start                       Event                                          Target                                Action                              Guard
        //---------+-------------------+----------------------------------------------------+-----------------------------------+-------------------------------------------+-----------------+
            row<            CNN,                    Change_model_CNNtoCNN_Evt,                          CNN,                             &m::CNN_Act                           /*none*/ >,
            row<            BNN,                    Change_model_BNNtoBNN_Evt,                          BNN,                             &m::BNN_Act                           /*none*/ >,
            row<            LSVM,                    Change_model_LSVMtoLSVM_Evt,                       LSVM,                            &m::LSVM_Act                          /*none*/ >,
            row<            CNN,                    Change_model_CNNtoBNN_Evt,                          BNN,                             &m::BNN_Act                           /*none*/ >,
            row<            CNN,                    Change_model_CNNtoLSVM_Evt,                         LSVM,                            &m::LSVM_Act                          /*none*/ >,
            row<            BNN,                    Change_model_BNNtoCNN_Evt,                          CNN,                             &m::CNN_Act                           /*none*/ >,
            row<            BNN,                    Change_model_BNNtoLSVM_Evt,                         LSVM,                            &m::LSVM_Act                          /*none*/ >,
            row<            LSVM,                   Change_model_LSVMtoCNN_Evt,                         CNN,                             &m::CNN_Act                           /*none*/ >,
            row<            LSVM,                   Change_model_LSVMtoBNN_Evt,                         BNN,                             &m::BNN_Act                           /*none*/ >>;
};
