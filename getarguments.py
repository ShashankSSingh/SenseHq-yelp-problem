import utils
import argparse

def get_arguments():
    """Obtain the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("City", type=str, help="City Name")

    args = parser.parse_args()
    return args

args = get_arguments()

file_city_name_list = utils.file_city_name
city_name_list = utils.city_name

if(args.City in city_name_list):
    print("City Found!! Continuing...")
    city_name = city_name_list[city_name_list.index(args.City)]
    file_city_name = file_city_name_list[city_name_list.index(args.City)]
else:
    print("City Not Found")
    exit()
