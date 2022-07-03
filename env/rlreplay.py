"""
Explanation:
-----------
- Epsiode is used by TradingEnvironment
- rlreplay Based on replay with some minor changes
- __iter__: make episode iterable to call next with step methods
- __len__ to return the episode length
...
"""


# use relative imports for other modules 
#from env.market import MarketState, Order, Trade
#from gym_linkage.tradingenv_v8 import MarketState, Order, Trade

# general imports
import copy
import datetime
import logging
logging.basicConfig(level=logging.CRITICAL) # logging.basicConfig(level=logging.NOTSET)
import os
import pandas as pd
import random
random.seed(42)
import time

# note: path goes to different project which already contained the data
SOURCE_DIRECTORY = "/Users/florianewald/PycharmProjects/l2-backtest-engine-v2X/efn2_backtesting"
DATETIME = "TIMESTAMP_UTC"

class Episode:

    def __init__(self,
        identifier_list:list,
        episode_start_buffer:str,
        episode_start:str,
        episode_end:str,
    ):
        """
        Prepare a single episode as a generator. The episode is the main 
        building block of each backtest. 

        :param identifier_list:
            pd.Timestamp, start building the market state, ignore agent

        :param episode_start_buffer:
            str, timestamp from which to start building the market state, ignore agent
        :param episode_start:
            str, timestamp from which to start informing the agent
        :param episode_end:
            str, timestamp from which to stop informing the agent
        """

        # data settings
        self.identifier_list = identifier_list

        # ...
        self._episode_start_buffer = pd.Timestamp(episode_start_buffer)
        self._episode_start = pd.Timestamp(episode_start)
        self._episode_end = pd.Timestamp(episode_end)

        # maximum deviation from the expected episode length
        # setup routine
        self._episode_setup(
            max_deviation_tol=300, # in seconds
        ) 

        # set buffering to true when initializing the episode
        # buffering will be set False in __next__() method
        self._episode_buffering = True

        # reset step counter to 0
        # steps are counted in the __next__() method
        self.step = 0

    # static attributes ---

    @property
    def episode_start_buffer(self): 
        return self._episode_start_buffer 
    
    @property
    def episode_start(self):
        return self._episode_start 

    @property
    def episode_end(self):
        return self._episode_end

    # dynamic attributes ---

    @property
    def episode_buffering(self):
        return self._episode_buffering

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def timestamp_next(self):
        return self._timestamp_next

    # episode setup ---

    # DONE
    def _episode_setup(self, max_deviation_tol=300):
        """
        Load and prepare episode for the  

        :param max_deviation_tol:
            int, maximum deviation from the expected episode length (in seconds)
        """

        # display progress ---

        # info
        logging.info("(INFO) episode from {episode_start} ({episode_start_buffer}) to {episode_end} is being prepared ...".format(
            episode_start=self._episode_start,
            episode_start_buffer=self._episode_start_buffer.time(),
            episode_end=self._episode_end,
        ))

        # in the beginning, unset flag to disallow iteration via __iter__ method
        self._episode_available = False

        # prepare data ---

        # build path_store to host all paths to load data from (only for this particular episode)
        path_store = self._build_path_store(self._episode_start, self._episode_end)
        # build data_store to host all data (only for this particular episode)
        data_store = self._build_data_store(self._episode_start, self._episode_end, path_store)
        # align data_store so that each data source has equal length
        data_store = self._align_data_store(data_store)
        # build data_monitor to iterate over
        data_monitor = self._build_data_monitor(data_store)


        # set attributes ---

        # TODO: can I use the __next__ to get to next _data_store and _data_monitor?
        # set data_store to iterate over using the __iter__ method
        self._data_store = data_store
        # set data_monitor to iterate over using the __iter__ method
        self._data_monitor = data_monitor

        # sanity check ---

        # total time_delta should not deviate from episode_length by more than <tolerance> seconds
        time_delta_observed = (
            abs(self._data_monitor.iloc[0, 0] - self._episode_start) +
            abs(self._data_monitor.iloc[-1, 0] - self._episode_end)
        )
        # ...
        time_delta_required = pd.Timedelta(max_deviation_tol, "s")
       
        # ... 
        assert time_delta_observed < time_delta_required, \
            "(ERROR) time delta exceeded max deviation tolerance (required: {required}, observed: {observed})".format(
                required=time_delta_required, 
                observed=time_delta_observed, 
            )
        
        # allow iteration ---

        # set flag to allow iteration via __iter__method
        self._episode_available = True

        # info
        logging.info("(INFO) episode has successfully been set and includes a total of {num_steps} steps".format(
            num_steps=len(data_monitor.index),
        ))

    # helper methods ---

    def _build_path_store(self, timestamp_start, timestamp_end):
        """
        Find paths and store them in the path_store dictionary together with 
        their corresponding key.
        
        Note that timestamp_start and timestamp_end must belong to the same 
        date!

        :param timestamp_start:
            pd.Timestamp, ...
        :param timestamp_end:
            pd.Timestamp, ...

        :return path_store:
            dict, {<identifier>: <path>, *}
        """

        path_store = dict()

        # ...
        assert timestamp_start.date() == timestamp_end.date(), \
            "(ERROR) timestamp_start and timestamp_end must belong to the same date"
        date = timestamp_start.date()
        date_string = str(date).replace("-", "")

        # path_list includes all paths available in directory
        path_list = [os.path.join(pre, file) for pre, _, sub in os.walk(SOURCE_DIRECTORY)
            for file in sub if not file.startswith((".", "_"))
        ]

        # ...
        for identifier in self.identifier_list:

            # identify matching criteria
            market_id, event_id = identifier.split(".")

            # make copy of path_list
            path_list_ = path_list.copy()
            # filter based on matching criteria
            path_list_ = filter(
                lambda path: market_id.lower() in path.lower(), path_list_)
            path_list_ = filter(
                lambda path: event_id.lower() in path.lower(), path_list_)
            path_list_ = filter(
                lambda path: date_string in path, path_list_)
            # ...
            path_list_ = list(path_list_)

            # if path_list_this is empty, raise Exception that is caught in calling method
            if not len(path_list_) == 1:
                raise Exception("(ERROR) could not find path for {identifier} between {timestamp_start} and {timestamp_end}".format(
                    identifier=identifier,
                    timestamp_start=timestamp_start, timestamp_end=timestamp_end,
                ))

            # there should be exactly one matching path
            path = path_list_[0]

            # add dataframe to output dictionary
            path_store[identifier] = path

        # info
        logging.info("(INFO) path_store has been built")

        return path_store

    def _build_data_store(self, timestamp_start, timestamp_end, path_store):
        """
        Load .csv(.gz) and .json files into dataframes and store them in the
        data_store dictionary together with their corresponding key.

        :param path_store:
            dict, {<identifier>: <path>, *}
        :param timestamp_start:
            pd.Timestamp, ...
        :param timestamp_start:
            pd.Timestamp, ...

        :return data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, original timestamps
        """

        data_store = dict()

        # ...
        for identifier in self.identifier_list:

            # load event_id 'BOOK' as .csv(.gz)
            if "BOOK" in identifier:
                df = pd.read_csv(path_store[identifier], parse_dates=[DATETIME])
            # load event_id 'TRADES' as .json
            if "TRADES" in identifier:
                df = pd.read_json(path_store[identifier], convert_dates=True)

            # if dataframe is empty, raise Exception that is caught in calling method
            if not len(df.index) > 0:
                raise Exception("(ERROR) could not find data for {identifier} between {timestamp_start} and {timestamp_end}".format(
                    identifier=identifier,
                    timestamp_start=timestamp_start, timestamp_end=timestamp_end,
                ))

            # make timestamp timezone-unaware
            df[DATETIME] = pd.DatetimeIndex(df[DATETIME]).tz_localize(None)
            # filter dataframe to include only rows with timestamp between timestamp_start and timestamp_end
            df = df[df[DATETIME].between(timestamp_start, timestamp_end)]

            # add dataframe to output dictionary
            data_store[identifier] = df

        # info
        logging.info("(INFO) data_store has been built")

        return data_store

    def _align_data_store(self, data_store):
        """
        Consolidate and split again all sources so that each source dataframe
        contains a state for each occurring timestamp across all sources.

        :param data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, original timestamps

        :return data_store:
            dict, {<identifier>: <pd.DataFrame>, *}, aligned timestamps
        """

        # unpack dictionary
        id_list, df_list = zip(*data_store.items())

        # rename columns and use id as prefix, exclude timestamp
        add_prefix = lambda id, df: df.rename(columns={x: f"{id}__{x}"
            for x in df.columns[1:]
        })
        df_list = list(map(add_prefix, id_list, df_list))

        # join df_list into df_merged (full outer join)
        df_merged = pd.concat([
            df.set_index(DATETIME) for df in df_list
        ], axis=1, join="outer").reset_index()

        # split df_merged into original df_list (all df are of equal length)
        df_list = [pd.concat([
            df_merged[[DATETIME]], # global timestamp
            df_merged[[x for x in df_merged.columns if id in x] # filtered by identifier
        ]], axis=1) for id in id_list]

        # rename columns and remove prefix, exclude timestamp
        del_prefix = lambda df: df.rename(columns={x: x.split("__")[1]
            for x in df.columns[1:]
        })
        df_list = list(map(del_prefix, df_list))

        # pack dictionary
        data_store = dict(zip(id_list, df_list))

        # info
        logging.info("(INFO) data_store has been aligned")

        return data_store

    def _build_data_monitor(self, data_store):
        """
        In addition to the sources dict, return a monitor dataframe that keeps
        track of changes in state across all sources.

        :param data_store:
            dict, {<source_id>: <pd.DataFrame>, *}, aligned timestamp

        :return data_monitor:
            pd.DataFrame, changes per source and timestamp
        """

        # setup dictionary based on timestamp
        datetime_index = list(data_store.values())[0][DATETIME]
        data_monitor = {DATETIME: datetime_index}

        # track changes per source and timestamp in series
        for key, df in data_store.items():
            data_monitor[key] = ~ df.iloc[:, 1:].isna().all(axis=1)


        # build monitor as dataframe from series
        data_monitor = pd.DataFrame(data_monitor)

        # info
        logging.info("(INFO) data_monitor has been built")

        return data_monitor

    def __len__(self):
        '''
        Returns length of the current episode, relevant for training loops
        over episodes and the done-flag of the RL environment.
        :return: current_episode_length
            int, length of episode
        '''
        current_episode_length = len(self._data_monitor)

        return current_episode_length


    #original __iter__ method:
    def __iter__(self):
        '''
        Iterate over the set episode. 
        
        NOTE: Use the self._episode_buffering flag to check if the buffering 
        phase has ended - only then should the agent be notified about market 
        updates. 
        '''
        
        # return, that is, disallow iteration if no episode has been set
        if not self._episode_available:
            return
        
        # ...
        logging.info("(INFO) episode has started ...")
        
        # set buffer flag
        self._episode_buffering = True

        # time
        time_start = time.time()

        # ...
        for step, timestamp, *monitor_state in self._data_monitor.itertuples():

            # update timestamps ---

            # track this timestamp
            self._timestamp = self._data_monitor.iloc[step, 0]
            
            # track next timestamp, prevent IndexError that would arise with the last step
            self._timestamp_next = self._data_monitor.iloc[min(
                step + 1, len(self._data_monitor.index) - 1
            ), 0]

            # display progress ---

            # ...
            progress = timestamp.value / (self._episode_end.value - self._episode_start_buffer.value)
            eta = (time.time() - time_start) / progress

            # info
            logging.info("(INFO) step {step}, progress {progress}, eta {eta}".format(
                step=step,
                progress=progress,
                eta=eta,
            ))

            # handle buffer phase ---
            
            # update buffer flag, agent should start being informed only after buffering phase has ended
            cache_episode_buffering = self._episode_buffering
            self._episode_buffering = timestamp < self._episode_start
            
            # info
            if cache_episode_buffering != self._episode_buffering:
                logging.info("(INFO) buffering phase for this episode has ended, allow trading ...")
            
            # find data ---

            # get identifier (column name) per updated source (based on self._data_monitor)
            identifier_list = (self._data_monitor
                .iloc[:, 1:]
                .columns[monitor_state]
                .values
            )

            # get data per updated source (based on self._data_store)
            data_list = [self._data_store[identifier].iloc[step, :] 
                for identifier in identifier_list
            ]

            # yield data ---

            # for each step, yield update via dictionary
            update = dict(zip(identifier_list, data_list)) # {<identifier>: <data>, *}

            # ...
            yield update
        
        # time
        time_end = time.time()
        
        # ...
        time_delta = round(time_end - time_start, 3)
        time_per_step = round((time_end - time_start) / step, 3)

        # info
        logging.info("(INFO)... episode has ended, took {time_delta}s for {step} steps ({time_per_step}s/step)".format(
            time_delta=time_delta,
            step=step,
            time_per_step=time_per_step,
        ))

    def __next__(self):
        '''
        Returns next book or trade update and counts step.
        :return update
            dict, contains identifier list and data updates.
        '''

        # return, that is, disallow iteration if no episode has been set
        if not self._episode_available:
            return

        # extract timestamp and *monitor_state from self._data_monitor
        # monitor state is a list of length 2 with booleans
        timestamp, *monitor_state = self._data_monitor.loc[self.step]

        # update timestamps ---

        # track this timestamp
        self._timestamp = self._data_monitor.iloc[self.step, 0]

        # track next timestamp, prevent IndexError that would arise with the last step
        self._timestamp_next = self._data_monitor.iloc[min(
            self.step + 1, len(self._data_monitor.index) - 1
        ), 0]

        # handle buffer phase ---


        # _episode_buffering will be set to false as soon as timestamp > _episode_start
        self._episode_buffering = timestamp < self._episode_start

        # find data ---

        # get identifier (column name) per updated source (based on self._data_monitor)
        identifier_list = (self._data_monitor
                           .iloc[:, 1:]
                           .columns[monitor_state]
                           .values
                           )

        # get data per updated source (based on self._data_store)
        data_list = [self._data_store[identifier].iloc[self.step, :]
                     for identifier in identifier_list
                     ]

        # return data ---

        # return update via dictionary
        update = dict(zip(identifier_list, data_list))  # {<identifier>: <data>, *}

        # count the step
        self.step = self.step + 1
        # return update
        return update



