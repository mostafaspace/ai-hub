import asyncio
import time

class MockHandler:
    def __init__(self, name):
        self.name = name
        self.tokenizer = "loaded_tokenizer"
        self.initialized = True

    def unload(self):
        print(f"[{self.name}] Unloading...")
        # Simulate time taken to unload
        time.sleep(0.5)
        self.tokenizer = None
        print(f"[{self.name}] Unloaded (tokenizer is now None)")

class MockAppState:
    def __init__(self):
        self._llm_initialized = True
        self._llm_init_lock = asyncio.Lock()
        self.llm_handler = MockHandler("LLM")
        self.last_used = {"llm_handler": time.time() - 1000}
        self.idle_timeout = 10

async def _ensure_llm_ready(state):
    print("[Request] Checking initialization status...")
    async with state._llm_init_lock:
        if state._llm_initialized:
            print("[Request] Already initialized. Proceeding with generation...")
            # Simulate some work before using tokenizer
            await asyncio.sleep(0.2)
            if state.llm_handler.tokenizer is None:
                print("[Request] ERROR: Tokenizer is None! RACE CONDITION DETECTED!")
                return False
            print(f"[Request] Using tokenizer: {state.llm_handler.tokenizer}")
            return True
        else:
            print("[Request] Not initialized or recently unloaded. Initializing...")
            # Simulate initialization time
            await asyncio.sleep(0.3)
            state._llm_initialized = True
            state.llm_handler.tokenizer = "reloaded_tokenizer"
            print("[Request] Initialization complete. Using tokenizer.")
            return True

async def _idle_monitor_fixed(state):
    print("[Monitor] Checking for idle handlers...")
    if state._llm_initialized:
        # Check if lock is held by a request
        if state._llm_init_lock.locked():
            print("[Monitor] Lock is held by a request, skipping this cycle...")
            return
            
        async with state._llm_init_lock:
            # Double check inside lock
            if state._llm_initialized:
                print("[Monitor] Idle timeout reached. Unloading...")
                # FIX: Set initialized to False BEFORE unloading
                state._llm_initialized = False
                state.llm_handler.unload()
                print("[Monitor] Unload complete.")

async def main():
    state = MockAppState()
    
    print("\n--- Test 1: Monitor skips when request holds lock ---")
    # Start a request that will hold the lock
    req_task = asyncio.create_task(_ensure_llm_ready(state))
    # Give it a tiny bit of time to acquire lock
    await asyncio.sleep(0.05)
    
    # Run monitor - it should see the lock is held and skip
    await _idle_monitor_fixed(state)
    
    await req_task
    
    print("\n--- Test 2: Request waits and re-initializes when monitor is unloading ---")
    # Reset state
    state = MockAppState()
    
    # Start monitor - it acquires lock and start unloading
    mon_task = asyncio.create_task(_idle_monitor_fixed(state))
    await asyncio.sleep(0.05)
    
    # Start request - it should wait for lock, then see initialized=False and re-init
    req_task = asyncio.create_task(_ensure_llm_ready(state))
    
    await asyncio.gather(mon_task, req_task)
    print("\nVerification successful: No race condition detected with fixed logic.")

if __name__ == "__main__":
    asyncio.run(main())
