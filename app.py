import sys
from neuralnet import *

#TODO
# better input normalization
# regularization
# gradient descent with adam optimization
# learning rate decay
# batch normalization
# mini batches

def execute_command(nn):
    command = input(">> ")

    if(command == "view"):
        view_command(nn)

    elif command == "build":
        build_command(nn)

    elif command == "train":
        train_command(nn)

    elif command == "predict":
        predict_command(nn)

    elif command == "save":
        save_command(nn)

    elif command  == "load":
        load_command(nn)

    elif command == "help":
        help_command()

    elif command == "quit":
        sys.exit()

    else:
        print("Command not recognized")

def view_command(nn):
    print("Current build")
    print(f"learning rate: {nn.learning_rate}")
    print(f"iterations: {nn.iterations}")
    print("layer structure: " + str(nn.layer_dims))

def build_command(nn):
    dims = input("Enter new layer dimensions of neural net: ")
    nn.layer_dims = [nn.train_x.shape[0]] + list(map(int, dims.split()))
    nn.learning_rate = float(input("Enter new learning rate: "))
    nn.iterations = int(input("Enter new number of training iterations: "))
    nn.W, nn.b = initialize_parameters(nn.layer_dims)
    print("Model built with specified parameters.")

def train_command(nn):
    train_x = nn.train_x
    train_y = nn.train_y
    layer_dims = nn.layer_dims
    learning_rate = nn.learning_rate
    iterations = nn.iterations

    nn.W, nn.b = model(train_x, train_y, layer_dims, learning_rate, iterations)
    accuracy_train = predict(nn.train_x, nn.train_y, nn.W, nn.b)
    accuracy_test = predict(nn.test_x, nn.test_y, nn.W, nn.b)
    print(f"Accuracy for training set: {accuracy_train}")
    print(f"Accuracy for test set: {accuracy_test}")

def predict_command(nn):
    image = input("Enter name of image: ")
    prediction = predict_image(nn.W, nn.b, image)
    print(f"model predicts: {prediction}")

def save_command(nn):
    save_parameters(nn.W, nn.b, nn.layer_dims, nn.learning_rate, nn.iterations)
    print("Build saved.")

def load_command(nn):
    W, b, layer_dims, learning_rate, iterations = load_parameters()

    nn.W = W
    nn.b = b
    nn.layer_dims = layer_dims
    nn.learning_rate = learning_rate
    nn.iterations = iterations

    print("Build loaded.")

def help_command():
    with open("save/help.txt", "r") as help_file:
        print(help_file.read())

def main():
    train_x, train_y, test_x, test_y = load_data()
    nn = NeuralNet(train_x, train_y, test_x, test_y)
    nn.W, nn.b = initialize_parameters(nn.layer_dims)

    print("Neural net for image recognition")
    print("Enter a command or type 'help' for more information.")

    while True:
        try:
            execute_command(nn)
        except ValueError:
            print("Wrong argument type for command.")
        except FileNotFoundError:
            print("File not found.")

if __name__ == "__main__":
    main()
