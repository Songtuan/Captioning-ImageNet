import collections

class State:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Condition:
    '''
    Condition class used to determine whether a specific condition is satisfied
    when a trigger event which may cause state transition happen. This class must
    be override unless no such condition exit
    '''

    def __init__(self):
        pass

    def update(self):
        pass

    def _eval(self, event):
        '''
        evaluate whether the input comes along with the trigger object satisfy
        this condition

        Args:
            event: an Event object which contain the input information
        '''
        assert 0, 'not implement error'
        pass


class Event:
    '''
    Overwrite this Event class to define an event which can trigger
    the state transition, e.g. the state machine receive an input token.
    The event object should also contain some information, e.g. the
    input token
    '''
    pass


class TransitionTable:
    '''
    TransitionTable store the relationship between (state, event object) and
    (condition, next_state). Particularly, it map current state along with the
    current trigger event to the corresponding condition and output state. If
    the condition is satisfied, the state transit to next_state
    '''

    def __init__(self, table=None):
        if table:
            self.__table = table
        else:
            self.__table = collections.defaultdict(list)

    def add(self, source, dest, event, condition=None):
        self.__table[(source, event)].append((condition, dest))

    def get(self, source, event):
        return self.__table[(source, event)] if (source, event) in self.__table else None


class StateMachine:
    def __init__(self, states=None, table=None, init_state=None, events=None):
        '''
        state_machine class control the main logic of the state transition

        Args:
            states: the dict of all states contained in state machine
                    with format {state_name: state object}
                    type: dict
            table: the instance of TransitionTable
                   type: TransitionTable
            init_state: the initial state's name of state machine
                        type: State object
            events: the dict which map event name and event object
                    type: dict
        '''
        self.states = states if states is not None else {}

        self.table = table if table is not None else TransitionTable()

        self.events = events if events is not None else None

        if init_state is not None and init_state not in self.states:
            self.states[init_state] = State(init_state)

        self.state = self.states[init_state] if init_state is not None else None

    def run(self, event):
        '''
        run the state machine to get next state base on the event

        Args:
            event: event object
        '''
        assert self.state is not None, 'current state is empty, try to use _set first'

        if self.table.get(self.state, event) is None:
            return

        if self.table.get(self.state, event) is None:
            return

        for condition, next_state in self.table.get(self.state, event):
            if condition is None or condition._eval(event):
                self.state = next_state
                return

        print('warning: no condition match')

    def get_event(self, name):
        assert name in self.events, 'invalid event, use _register() to registe a new event'

        return self.events[name]

    def get(self):
        '''
        return the current state of state machine
        '''
        return self.state

    def set_state(self, name):
        '''
        set the current state of state machine
        Args:
            name: the desired state's name
        '''
        self.state = self.states[name]

    def get_states(self):
        return self.states

    def get_transition(self, source_name, target_name, event_name):
        assert source_name in self.states, 'invalid state name, use _add_state() to add a new state first'
        assert target_name in self.states, 'invalid state name, use _add_state() to add a new state first'
        assert event_name in self.events, 'invalid event, use _register() to regist a new event'

        source = self.states[source_name]
        target = self.states[target_name]
        event = self.events[event_name]

        if self.table.get(source, event) is None:
            # print('warning {}, {} pair not in transition table'.format(source_name, event_name))
            return None

        for condition, dest in self.table.get(source, event):
            if target == dest:
                return condition

        #print('Warnning: No matched condition found')
        return None

    def add_transition(self, source_name, dest_name, event_name, condition=None):
        '''
        add a new entry to transition table

        Args:
            source_name: source state's name
                         type: str
            dest_name: destination state's name
                       type: str
            event_name: trigger event's name
                        type: str
            condition: condition need to be satisfied
                       type: Condition object
        '''
        assert source_name in self.states, 'invalid state name, use _add_state() to add a new state first'
        assert dest_name in self.states, 'invalid state name, use _add_state() to add a new state first'
        assert event_name in self.events, 'invalid event, use _register() to regist a new event'

        source = self.states[source_name]
        dest = self.states[dest_name]
        event = self.events[event_name]

        self.table.add(source, dest, event, condition)

    def add_state(self, name, state=None):
        '''
        add a new state to state machine

        Args:
            name: the name of the state
                  type: str
            state: State object
                   type: State

        Returns:
            Boolean: True if successfully add state, otherwise False
        '''
        if name in self.states:
            return False
        self.states[name] = state if state is not None else State(name)
        return True

    def regist(self, name, event):
        '''
        regist trigger event which may cause state transition
        Args:
            name: the event's name
                  type: str
            event: the instance of Event class
                   type: Event
        '''
        self.events[name] = event

    def add_state_idx_mapping(self, state_idx_mapping):
        self.state_idx_mapping = state_idx_mapping
