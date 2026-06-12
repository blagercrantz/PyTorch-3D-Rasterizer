from pynput import keyboard

class InputManager:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        if hasattr(key, 'char') and key.char:
            self.pressed_keys.add(key.char)
        else:
            self.pressed_keys.add(key)

    def _on_release(self, key):
        if hasattr(key, 'char') and key.char:
            self.pressed_keys.discard(key.char)
        else:
            self.pressed_keys.discard(key)

    def is_held(self, key_name):
        if key_name in self.pressed_keys:
            return True
        key_map = {'up': keyboard.Key.up, 'down': keyboard.Key.down, 'left': keyboard.Key.left, 'right': keyboard.Key.right}
        return key_map.get(key_name) in self.pressed_keys