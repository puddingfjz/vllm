from multiprocessing import Process, Array, set_start_method, get_start_method, Event

# def child_process(shared_array):
#     try:
#         # Modify the shared array
#         # for i in range(len(shared_array)):
#         #     shared_array[i] += 1
#         shared_array.set()
#     except Exception as e:
#         print(f"Exception in child process: {e}")

# def main():
#     try:
#         # Set the start method to 'spawn' if it's not already set
#         if get_start_method(allow_none=True) != 'spawn':
#             set_start_method('spawn', force=True)

#         # Create a shared array with initial values
#         shared_array = Array('i', [1, 2, 3, 4, 5])
#         event = Event()

#         # Create a subprocess
#         p = Process(target=child_process, args=(event,))
#         p.start()
#         p.join()

#         # Check the results
#         print("Modified array:", event.is_set())
#     except RuntimeError as e:
#         print(f"RuntimeError: {e}")
#     except Exception as e:
#         print(f"Exception: {e}")

# if __name__ == "__main__":
#     main()





import multiprocessing
import time

def child_process(shared_array):
    try:
        # Modify the shared array
        for i in range(len(shared_array)):
            shared_array[i] += 1
        # shared_array.set()
    except Exception as e:
        print(f"Exception in child process: {e}")

def main():
    try:
        # Set the start method to 'spawn' if it's not already set
        if get_start_method(allow_none=True) != 'spawn':
            set_start_method('spawn', force=True)

        # Create a shared array with initial values
        shared_array = Array('i', [1, 2, 3, 4, 5])
        event = Event()

        # Create a subprocess
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=1) as pool:
            # List of numbers to square
            # numbers = [1, 2, 3, 4, 5]

            # Map the function to the pool
            results = pool.map(child_process, [shared_array])

        # Check the results
        print("Modified array:", list(shared_array))
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    except Exception as e:
        print(f"Exception: {e}")




if __name__ == "__main__":
    main()
