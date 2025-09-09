#include "fsm.h"
#include "FSMHelper.h"
#include "mainhelper.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

int main() {
    UI_Model UI_Model;
    std::string receivedData;

    std::cout << "Current state: " << UI_Model.GetCurrentState() << std::endl;

    while (1) {
        std::string current_page = data_change_model();
        std::cout << "Request model " << current_page << std::endl;
        Process_model_Changeloop(current_page);
    }
    return 0;
}
