import yaml
import random
import heapq
import time

class ListRandomGenerator:
    """List-based random number generator, mimicking the Java class AmostradorEmLista"""
    def __init__(self, numbers):
        self.numbers = iter(numbers)

    def next(self):
        try:
            return next(self.numbers)
        except StopIteration:
            raise Exception("OutOfNumbers: Reached the limit of random numbers!")

class LinearCongruentialGenerator:
    """Linear Congruential Generator (LCG) similar to the Java class AmostradorCongruenteLinear"""
    def __init__(self, limit, seed=1):
        self.a = 25214903917
        self.c = 11
        self.m = 281474976710656  # 2^48
        self.x = seed
        self.limit = limit
        self.count = 0

    def next(self):
        if self.count >= self.limit:
            raise Exception("OutOfNumbers: Reached the limit of random numbers!")
        self.x = (self.a * self.x + self.c) % self.m
        self.count += 1
        return self.x / self.m

class Queue:
    """Queue class simulating the behavior of Fila.java"""
    def __init__(self, id, capacity, servers, min_arrival, max_arrival, min_service, max_service):
        self.id = id
        self.capacity = capacity
        self.servers = servers
        self.min_arrival = min_arrival
        self.max_arrival = max_arrival
        self.min_service = min_service
        self.max_service = max_service
        self.population = 0
        self.lost = 0
        self.total_time = 0.0
        self.destinations = []
        self.busy_servers = 0  # Number of servers currently busy
        self.waiting_line = []  # List of customers waiting (arrival times)

    def add_destination(self, destination, probability):
        self.destinations.append((destination, probability))

    def next_destination(self, random_number):
        cumulative = 0
        for dest, prob in self.destinations:
            cumulative += prob
            if random_number < cumulative:
                return dest
        return None

    def __repr__(self):
        return f"Queue({self.id}, pop={self.population}, lost={self.lost})"

class Event:
    """Event class representing events in the simulation (e.g., arrival, departure)"""
    def __init__(self, time, event_type, queue, destination=None, arrival_time=None):
        self.time = time  # Time at which the event occurs
        self.event_type = event_type
        self.queue = queue
        self.destination = destination
        self.arrival_time = arrival_time  # Time when the customer arrived at the queue

    def __lt__(self, other):
        return self.time < other.time

class SimulationEnvironment:
    def __init__(self):
        self.time = 0.0
        self.global_time = 0.0
        self.last_event_time = 0.0
        self.events = []
        self.queues = {}
        self.random_generator = None
        self.random_number_count = 0
        self.max_random_numbers = 100000  # Limit to 100,000 random numbers

    def add_queue(self, queue):
        self.queues[queue.id] = queue

    def schedule_event(self, event):
        heapq.heappush(self.events, event)

    def run(self, random_generator):
        self.random_generator = random_generator
        start_time = time.time()

        try:
            while self.events:
                current_event = heapq.heappop(self.events)
                if self.random_number_count >= self.max_random_numbers:
                    print(f"Simulation stopped after using {self.max_random_numbers} random numbers.")
                    break

                # Update global time
                self.global_time = current_event.time
                self.time = current_event.time

                if current_event.event_type == 'arrival':
                    self.handle_arrival(current_event)
                elif current_event.event_type == 'departure':
                    self.handle_departure(current_event)
        except Exception as e:
            if str(e) == "OutOfNumbers: Reached the limit of random numbers!":
                print("Random number limit reached. Ending simulation.")
            else:
                raise

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tempo total de simulação: {self.global_time:.2f} UT")
        print(f"Tempo real de execução: {execution_time:.4f} segundos")
        self.print_metrics()

    def print_metrics(self):
        """Print the final metrics for the simulation."""
        print("\n=== Simulation Metrics ===")
        print(f"Global Time: {self.global_time:.2f} UT")

        for queue in self.queues.values():
            print(f"\nQueue {queue.id}:")
            print(f"  - Population: {queue.population}")
            print(f"  - Lost Customers: {queue.lost}")
            print(f"  - Total Accumulated Time: {queue.total_time:.2f} UT")

        # Add routing probability report
        print("\n=== Routing Probability Distribution ===")
        for queue in self.queues.values():
            if queue.destinations:
                print(f"\nQueue {queue.id} Routing Probabilities:")
                for dest, prob in queue.destinations:
                    dest_id = 'exit' if dest == 'exit' else dest.id
                    print(f"  - To {dest_id}: {prob * 100:.2f}%")

    def handle_arrival(self, event):
        queue = event.queue
        if queue.capacity == -1 or queue.population < queue.capacity:
            queue.population += 1
            if queue.busy_servers < queue.servers:
                # Server is available
                queue.busy_servers += 1
                service_time = self.random_service_time(queue)
                departure_event = Event(self.time + service_time, 'departure', queue, arrival_time=self.time)
                self.schedule_event(departure_event)
            else:
                # Add customer to waiting line
                queue.waiting_line.append(self.time)
        else:
            queue.lost += 1

        # Schedule next arrival if random number limit not reached
        if self.random_number_count < self.max_random_numbers:
            arrival_time = self.random_arrival_time(queue)
            new_arrival = Event(self.time + arrival_time, 'arrival', queue)
            self.schedule_event(new_arrival)

    def handle_departure(self, event):
        queue = event.queue
        queue.population -= 1
        queue.busy_servers -= 1
        # Calculate time spent in queue (waiting + service)
        time_in_queue = self.time - event.arrival_time
        queue.total_time += time_in_queue
        # Check if there are customers waiting
        if queue.waiting_line:
            next_customer_arrival_time = queue.waiting_line.pop(0)
            queue.busy_servers += 1
            service_time = self.random_service_time(queue)
            departure_event = Event(self.time + service_time, 'departure', queue, arrival_time=next_customer_arrival_time)
            self.schedule_event(departure_event)
        random_number = self.random_generator.next()
        self.random_number_count += 1
        destination = queue.next_destination(random_number)
        if destination == 'exit':
            pass  # Customer exits the system
        elif destination:
            if self.random_number_count < self.max_random_numbers:
                arrival_time = self.random_arrival_time(destination)
                new_arrival = Event(self.time + arrival_time, 'arrival', destination)
                self.schedule_event(new_arrival)

    def random_arrival_time(self, queue):
        random_number = self.random_generator.next()
        self.random_number_count += 1
        # If minArrival and maxArrival are zero (no external arrivals), return zero
        if queue.min_arrival == 0.0 and queue.max_arrival == 0.0:
            return 0.0
        else:
            return (queue.max_arrival - queue.min_arrival) * random_number + queue.min_arrival

    def random_service_time(self, queue):
        random_number = self.random_generator.next()
        self.random_number_count += 1
        return (queue.max_service - queue.min_service) * random_number + queue.min_service

# Function to load the model
def load_model(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    env = SimulationEnvironment()
    for queue_id, queue_data in data['queues'].items():
        queue = Queue(
            queue_id,
            queue_data['capacity'],
            queue_data['servers'],
            queue_data['minArrival'],
            queue_data['maxArrival'],
            queue_data['minService'],
            queue_data['maxService']
        )
        env.add_queue(queue)

    for link in data['network']:
        source = env.queues[link['source']]
        target = link['target']

        if target == 'exit':
            source.add_destination('exit', link['probability'])
        else:
            destination = env.queues[target]
            source.add_destination(destination, link['probability'])

    # Initial customer arrival at time 2.0
    initial_queue = env.queues['fila1']
    initial_arrival = Event(2.0, 'arrival', initial_queue)
    env.schedule_event(initial_arrival)

    return env

# Main function
def run_simulation(model_file, seed_index=0, use_lcg=False):
    with open(model_file, 'r') as file:
        model_data = yaml.safe_load(file)

    seeds = model_data['seeds']

    if use_lcg:
        # Use Linear Congruential Generator (LCG) for random numbers
        random_generator = LinearCongruentialGenerator(limit=100000, seed=seeds[seed_index])  
    else:
        # Use List Random Generator with Python's random numbers
        random_numbers = [random.random() for _ in range(100000)]
        random_generator = ListRandomGenerator(random_numbers)

    env = load_model(model_file)
    env.run(random_generator)

# Run the simulation
if __name__ == '__main__':
    run_simulation('model.yml', seed_index=0, use_lcg=True)  # Use LCG by default