#!/usr/bin/env python3
"""
Enhanced Ableton MCP Remote Script - Extended Parameter Control
Supports tempo, master volume, track arming, clip launching, and more parameters
Compatible with Ableton Live 10+ (Python 3)
"""

import socket
import threading
import json
import time
import math

# Ableton Live API imports
import Live
from _Framework.ControlSurface import ControlSurface


class EnhancedAbletonMCP(ControlSurface):
    """Enhanced remote script with comprehensive parameter control"""

    def __init__(self, c_instance):
        """Initialize the Enhanced Remote Script"""
        ControlSurface.__init__(self, c_instance)

        # Core Ableton Live objects
        self.c_instance = c_instance
        self.song = c_instance.song()
        self.application = Live.Application.get_application()

        # Browser integration
        self.browser = self.application.browser
        self.device_cache = {}
        
        # Socket server for MCP communication
        self.socket_host = "127.0.0.1"
        self.socket_port = 9001
        self.socket_server = None
        self.client_socket = None
        self.server_thread = None
        self.is_running = False

        # Enhanced command handlers
        self.command_handlers = {
            # Basic commands
            "create_tracks": self.handle_create_tracks,
            "set_parameter": self.handle_set_parameter,
            "add_audio_effect": self.handle_add_audio_effect,
            "remove_audio_effect": self.handle_remove_audio_effect,
            "get_session_info": self.handle_get_session_info,
            "get_detailed_session_info": self.handle_get_detailed_session_info,
            "get_track_effects": self.handle_get_track_effects,
            "get_available_devices": self.handle_get_available_devices,
            "search_devices": self.handle_search_devices,
            
            # Track management
            "rename_track": self.handle_rename_track,
            "delete_track": self.handle_delete_track,
            # "delete_tracks": removed - handled by voice script calling delete_track multiple times
            "duplicate_track": self.handle_duplicate_track,
            "group_tracks": self.handle_group_tracks,
            "batch_commands": self.handle_batch_commands,  # New batch executor
            
            # Transport controls
            "transport_play": self.handle_transport_play,
            "transport_stop": self.handle_transport_stop,
            "transport_record": self.handle_transport_record,
            "set_tempo": self.handle_set_tempo,
            "set_time_signature": self.handle_set_time_signature,
            "set_loop": self.handle_set_loop,
            "toggle_metronome": self.handle_toggle_metronome,
            
            # Master/Global controls
            "set_master_volume": self.handle_set_master_volume,
            "set_cue_volume": self.handle_set_cue_volume,
            "set_crossfader": self.handle_set_crossfader,
            "set_groove_amount": self.handle_set_groove_amount,
            
            # Track controls
            "arm_track": self.handle_arm_track,
            "disarm_track": self.handle_disarm_track,
            "set_track_monitor": self.handle_set_track_monitor,
            "set_track_sends": self.handle_set_track_sends,
            
            # Clip controls
            "launch_clip": self.handle_launch_clip,
            "stop_clip": self.handle_stop_clip,
            "launch_scene": self.handle_launch_scene,
            "stop_all_clips": self.handle_stop_all_clips,
            
            # Browser controls
            "browse_devices": self.handle_browse_devices,
            "load_preset": self.handle_load_preset,
            
            # Legacy handlers
            "play": self.handle_transport_play,
            "stop": self.handle_transport_stop,
            "record": self.handle_transport_record,
        }

        # Track name to index mapping
        self.track_name_cache = {}
        self.update_track_cache()

        # Build device cache
        self.log_message("Building enhanced device cache...")
        self.build_enhanced_device_cache()
        self.log_message(f"Enhanced device cache built with {len(self.device_cache)} items")

        # Set up socket server
        self.start_socket_server()

        # Listen for song changes
        self.song.add_tracks_listener(self.update_track_cache)

        # Debug: Log available command handlers
        self.log_message(f"Available command handlers: {list(self.command_handlers.keys())}")

        self.log_message("Enhanced Ableton MCP Remote Script loaded successfully")

    def disconnect(self):
        """Clean shutdown"""
        self.is_running = False

        if self.song.tracks_has_listener(self.update_track_cache):
            self.song.remove_tracks_listener(self.update_track_cache)

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.socket_server:
            try:
                self.socket_server.close()
            except:
                pass

        ControlSurface.disconnect(self)
        self.log_message("Enhanced Remote Script disconnected")

    # ========== ENHANCED DEVICE CACHE ==========

    def build_enhanced_device_cache(self):
        """Build device cache with better search capabilities"""
        try:
            self.device_cache = {}
            self.log_message("Building enhanced device cache...")

            # Enhanced categories including more device types
            categories = [
                ("audio_effects", self.browser.audio_effects),
                ("instruments", self.browser.instruments),
                ("midi_effects", self.browser.midi_effects),
                ("drum_racks", getattr(self.browser, "drum_racks", None)),
                ("samples", getattr(self.browser, "samples", None))
            ]

            for category_name, category in categories:
                try:
                    if category:
                        self.log_message(f"Scanning {category_name}...")
                        self._index_enhanced_browser_items(category, category_name, [])
                        self.log_message(f"{category_name} done, cache now has {len(self.device_cache)} items")
                except Exception as e:
                    self.log_message(f"Error scanning {category_name}: {str(e)}")

            # Add common Ableton device mappings
            self._add_device_aliases()

            self.log_message(f"Enhanced device cache build complete with {len(self.device_cache)} items")

        except Exception as e:
            self.log_message(f"Enhanced cache build failed: {str(e)}")

    def _index_enhanced_browser_items(self, item, category, path):
        """Enhanced indexing with better device recognition"""
        try:
            if not item:
                return

            current_path = path + [str(item.name)]

            # Check if loadable
            try:
                if hasattr(item, "is_loadable") and item.is_loadable:
                    item_name = str(item.name).lower()
                    
                    self.device_cache[item_name] = {
                        "item": item,
                        "name": str(item.name),
                        "category": category,
                        "path": current_path,
                    }

                    # Add word-based search
                    words = item_name.replace("-", " ").replace("_", " ").split()
                    for word in words:
                        if len(word) > 2 and word not in self.device_cache:
                            self.device_cache[word] = {
                                "item": item,
                                "name": str(item.name),
                                "category": category,
                                "path": current_path,
                            }
            except:
                pass

            # Process children
            try:
                if hasattr(item, "children") and item.children:
                    for child in item.children:
                        self._index_enhanced_browser_items(child, category, current_path)
            except:
                pass

        except Exception as e:
            pass

    def _add_device_aliases(self):
        """Add common device name aliases for better recognition"""
        aliases = {
            'reverb': ['reverb'],
            'delay': ['delay', 'echo'],
            'compressor': ['compressor', 'comp'],
            'eq': ['eq eight', 'eq three'],
            'equalizer': ['eq eight'],
            'distortion': ['saturator', 'overdrive'],
            'filter': ['auto filter', 'filter'],
            'gate': ['gate'],
            'limiter': ['limiter'],
            'chorus': ['chorus'],
            'flanger': ['flanger'],
            'phaser': ['phaser'],
            'drum bus': ['drum bus'],  # This should help with "Drum Buss"
            'glue compressor': ['glue compressor'],
            'multiband dynamics': ['multiband dynamics'],
            'operator': ['operator'],
            'wavetable': ['wavetable'],
            'analog': ['analog'],
            'bass': ['bass'],
            'drums': ['drums', 'drum rack'],
            'impulse': ['impulse']
        }
        
        for alias, device_names in aliases.items():
            for device_name in device_names:
                for key, item_data in self.device_cache.items():
                    if device_name in key.lower():
                        if alias not in self.device_cache:
                            self.device_cache[alias] = item_data
                        break

    # ========== ENHANCED PARAMETER HANDLERS ==========

    def handle_set_parameter(self, command):
        """Enhanced parameter setting with more controls"""
        try:
            parameter = command.get("parameter")
            target = command.get("target", "")
            value = command.get("value")

            # Route to appropriate handler
            if parameter.startswith("mixer_"):
                return self.handle_mixer_parameter(parameter, target, value)
            elif parameter.startswith("transport_"):
                return self.handle_transport_parameter(parameter, value)
            elif parameter in ["master_volume", "tempo", "time_signature", "loop", "metronome"]:
                return self.handle_global_parameter(parameter, value)
            else:
                return {"error": f"Unknown parameter: {parameter}"}

        except Exception as e:
            return {"error": f"Failed to set parameter: {str(e)}"}

    def handle_global_parameter(self, parameter, value):
        """Handle global/master parameters"""
        try:
            if parameter == "master_volume":
                return self.handle_set_master_volume({"value": value})
            elif parameter == "tempo":
                return self.handle_set_tempo({"tempo": value})
            elif parameter == "time_signature":
                return self.handle_set_time_signature({"numerator": value[0], "denominator": value[1]})
            elif parameter == "loop":
                return self.handle_set_loop({"enabled": value})
            elif parameter == "metronome":
                return self.handle_toggle_metronome({"enabled": value})
            else:
                return {"error": f"Unknown global parameter: {parameter}"}
        except Exception as e:
            return {"error": f"Failed to set global parameter: {str(e)}"}

    def handle_transport_parameter(self, parameter, value):
        """Handle transport-specific parameters"""
        try:
            if parameter == "transport_tempo":
                return self.handle_set_tempo({"tempo": value})
            elif parameter == "transport_loop":
                return self.handle_set_loop({"enabled": value})
            elif parameter == "transport_metronome":
                return self.handle_toggle_metronome({"enabled": value})
            else:
                return {"error": f"Unknown transport parameter: {parameter}"}
        except Exception as e:
            return {"error": f"Failed to set transport parameter: {str(e)}"}

    # ========== NEW TRANSPORT CONTROLS ==========

    def handle_set_tempo(self, command):
        """Set the song tempo"""
        try:
            tempo = command.get("tempo")
            if tempo is None:
                return {"error": "No tempo value provided"}
            
            tempo = float(tempo)
            if 20 <= tempo <= 999:
                self.song.tempo = tempo
                return {"action": "set_tempo", "tempo": tempo, "status": "success"}
            else:
                return {"error": f"Tempo {tempo} out of range (20-999 BPM)"}
        except Exception as e:
            return {"error": f"Failed to set tempo: {str(e)}"}

    def handle_set_time_signature(self, command):
        """Set the time signature"""
        try:
            numerator = command.get("numerator", 4)
            denominator = command.get("denominator", 4)
            
            self.song.signature_numerator = int(numerator)
            self.song.signature_denominator = int(denominator)
            
            return {
                "action": "set_time_signature",
                "numerator": numerator,
                "denominator": denominator,
                "status": "success"
            }
        except Exception as e:
            return {"error": f"Failed to set time signature: {str(e)}"}

    def handle_set_loop(self, command):
        """Toggle or set loop on/off"""
        try:
            enabled = command.get("enabled")
            if enabled is None:
                # Toggle current state
                self.song.loop = not self.song.loop
            else:
                self.song.loop = bool(enabled)
            
            return {"action": "set_loop", "enabled": self.song.loop, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to set loop: {str(e)}"}

    def handle_toggle_metronome(self, command):
        """Toggle metronome on/off"""
        try:
            enabled = command.get("enabled")
            if enabled is None:
                # Toggle current state
                self.song.metronome = not self.song.metronome
            else:
                self.song.metronome = bool(enabled)
            
            return {"action": "toggle_metronome", "enabled": self.song.metronome, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to toggle metronome: {str(e)}"}

    # ========== MASTER/GLOBAL CONTROLS ==========

    def handle_set_master_volume(self, command):
        """Set the master volume"""
        try:
            volume = command.get("value")
            if volume is None:
                return {"error": "No volume value provided"}
            
            # Convert from percentage or dB to Live's linear scale
            if isinstance(volume, str) and volume.endswith('%'):
                volume = float(volume.rstrip('%')) / 100.0
            elif volume > 1.0:  # Assume percentage if > 1
                volume = volume / 100.0
            elif volume < 0:  # Assume dB if negative
                volume = pow(10, volume / 20.0)
            
            # Clamp to valid range
            volume = max(0.0, min(1.0, float(volume)))
            
            # Set master volume (this might need adjustment based on Live version)
            master_track = self.song.master_track
            master_track.mixer_device.volume.value = volume
            
            return {"action": "set_master_volume", "volume": volume, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to set master volume: {str(e)}"}

    def handle_set_cue_volume(self, command):
        """Set the cue/headphone volume"""
        try:
            volume = float(command.get("value", 0.5))
            volume = max(0.0, min(1.0, volume))
            
            # This might need adjustment based on Live's API
            if hasattr(self.song, "cue_volume"):
                self.song.cue_volume = volume
                
            return {"action": "set_cue_volume", "volume": volume, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to set cue volume: {str(e)}"}

    def handle_set_crossfader(self, command):
        """Set the crossfader position"""
        try:
            position = float(command.get("value", 0.0))
            position = max(-1.0, min(1.0, position))
            
            master_track = self.song.master_track
            if hasattr(master_track.mixer_device, "crossfader"):
                master_track.mixer_device.crossfader.value = position
                
            return {"action": "set_crossfader", "position": position, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to set crossfader: {str(e)}"}

    def handle_set_groove_amount(self, command):
        """Set the global groove amount"""
        try:
            amount = float(command.get("value", 0.0))
            amount = max(0.0, min(1.31, amount))  # 0-131%
            
            if hasattr(self.song, "groove_amount"):
                self.song.groove_amount = amount
                
            return {"action": "set_groove_amount", "amount": amount, "status": "success"}
        except Exception as e:
            return {"error": f"Failed to set groove amount: {str(e)}"}

    # ========== TRACK MANAGEMENT HANDLERS ==========

    def handle_rename_track(self, command):
        """Rename an existing track"""
        try:
            track_name = command.get("track", "selected")
            new_name = command.get("new_name", "")
            
            if not new_name:
                return {"error": "No new name provided"}
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            old_name = track.name
            track.name = new_name
            
            # Update track cache
            self.update_track_cache()
            
            return {
                "action": "rename_track",
                "old_name": old_name,
                "new_name": new_name,
                "status": "renamed"
            }
            
        except Exception as e:
            return {"error": f"Failed to rename track: {str(e)}"}

    def handle_delete_tracks(self, command):
        """Delete multiple tracks in sequence"""
        try:
            track_names = command.get("tracks", [])
            if not track_names:
                return {"error": "No tracks specified for deletion"}
            
            deleted_tracks = []
            failed_deletions = []
            
            # Process deletions from highest index to lowest to avoid index shifting issues
            tracks_to_delete = []
            for track_name in track_names:
                track = self.find_track_improved(track_name)
                if track:
                    tracks = list(self.song.tracks)
                    track_index = tracks.index(track)
                    tracks_to_delete.append((track_index, track.name, track))
                else:
                    failed_deletions.append(f"Track not found: {track_name}")
            
            # Sort by index in descending order
            tracks_to_delete.sort(reverse=True)
            
            # Delete tracks
            for track_index, track_name, track in tracks_to_delete:
                try:
                    self.song.delete_track(track_index)
                    deleted_tracks.append(track_name)
                    self.log_message(f"Deleted track: {track_name}")
                except Exception as e:
                    failed_deletions.append(f"Failed to delete {track_name}: {str(e)}")
            
            # Update track cache
            self.update_track_cache()
            
            return {
                "action": "delete_tracks",
                "deleted_tracks": deleted_tracks,
                "failed_deletions": failed_deletions,
                "deleted_count": len(deleted_tracks),
                "status": "completed"
            }
            
        except Exception as e:
            return {"error": f"Failed to delete tracks: {str(e)}"}

    def handle_batch_commands(self, command):
        """Execute multiple commands in sequence"""
        try:
            commands = command.get("commands", [])
            if not commands:
                return {"error": "No commands provided in batch"}
            
            results = []
            successful_count = 0
            
            for i, sub_command in enumerate(commands):
                try:
                    action = sub_command.get("action")
                    if action in self.command_handlers:
                        result = self.command_handlers[action](sub_command)
                        results.append({
                            "step": i + 1,
                            "action": action,
                            "result": result,
                            "status": "success" if "error" not in result else "failed"
                        })
                        
                        if "error" not in result:
                            successful_count += 1
                        
                        # Small delay between commands
                        time.sleep(0.2)
                    else:
                        results.append({
                            "step": i + 1,
                            "action": action,
                            "result": {"error": f"Unknown action: {action}"},
                            "status": "failed"
                        })
                        
                except Exception as e:
                    results.append({
                        "step": i + 1,
                        "action": sub_command.get("action", "unknown"),
                        "result": {"error": str(e)},
                        "status": "failed"
                    })
            
            # Update track cache after batch
            self.update_track_cache()
            
            return {
                "action": "batch_commands",
                "total_commands": len(commands),
                "successful_commands": successful_count,
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            return {"error": f"Failed to execute batch commands: {str(e)}"}

    def handle_delete_track(self, command):
        """Delete an existing track with better error handling"""
        try:
            track_name = command.get("track", "selected")
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            tracks = list(self.song.tracks)
            track_index = tracks.index(track)
            track_name_to_delete = track.name
            
            # Use a scheduled task for deletion to avoid blocking
            def delete_track_task():
                try:
                    self.song.delete_track(track_index)
                    self.update_track_cache()
                    self.log_message(f"Deleted track: {track_name_to_delete}")
                except Exception as e:
                    self.log_message(f"Error deleting track: {str(e)}")
            
            self.schedule_message(1, delete_track_task)
            
            return {
                "action": "delete_track",
                "deleted_track": track_name_to_delete,
                "track_index": track_index,
                "status": "deleting"
            }
            
        except Exception as e:
            return {"error": f"Failed to delete track: {str(e)}"}

    def handle_duplicate_track(self, command):
        """Duplicate an existing track"""
        try:
            track_name = command.get("track", "selected")
            new_name = command.get("new_name", None)
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            tracks = list(self.song.tracks)
            track_index = tracks.index(track)
            
            def duplicate_track_task():
                try:
                    # Duplicate the track
                    self.song.duplicate_track(track_index)
                    
                    # Get the duplicated track (should be right after original)
                    new_tracks = list(self.song.tracks)
                    duplicated_track = new_tracks[track_index + 1]
                    
                    # Set custom name if provided
                    if new_name:
                        duplicated_track.name = new_name
                    
                    # Update track cache
                    self.update_track_cache()
                    
                    self.log_message(f"Duplicated track '{track.name}' -> '{duplicated_track.name}'")
                    
                except Exception as e:
                    self.log_message(f"Error duplicating track: {str(e)}")
            
            self.schedule_message(1, duplicate_track_task)
            
            return {
                "action": "duplicate_track",
                "source_track": track.name,
                "new_name": new_name if new_name else f"{track.name} Copy",
                "status": "duplicating"
            }
            
        except Exception as e:
            return {"error": f"Failed to duplicate track: {str(e)}"}

    def handle_group_tracks(self, command):
        """Group selected tracks"""
        try:
            track_names = command.get("tracks", [])
            group_name = command.get("group_name", "Group")
            
            if not track_names:
                return {"error": "No tracks specified for grouping"}
            
            # Find all tracks to group
            tracks_to_group = []
            track_indices = []
            
            for track_name in track_names:
                track = self.find_track_improved(track_name)
                if track:
                    tracks_to_group.append(track)
                    tracks = list(self.song.tracks)
                    track_indices.append(tracks.index(track))
                else:
                    return {"error": f"Track not found: {track_name}"}
            
            # Sort indices to group in order
            track_indices.sort()
            
            def group_tracks_task():
                try:
                    # Create group track
                    if len(track_indices) > 1:
                        # Group the tracks (Ableton automatically creates group)
                        first_index = track_indices[0]
                        last_index = track_indices[-1]
                        
                        # Select tracks to group
                        for i, track_index in enumerate(track_indices):
                            if i == 0:
                                self.song.view.selected_track = list(self.song.tracks)[track_index]
                            else:
                                # Add to selection (this might need adjustment based on Live API)
                                pass
                        
                        # Create group (this creates a group track and moves selected tracks into it)
                        self.song.create_group_track()
                        
                        # Rename the group
                        tracks = list(self.song.tracks)
                        group_track = tracks[first_index]  # Group track takes the position of first track
                        group_track.name = group_name
                        
                        # Update track cache
                        self.update_track_cache()
                        
                        self.log_message(f"Grouped {len(track_names)} tracks into '{group_name}'")
                    
                except Exception as e:
                    self.log_message(f"Error grouping tracks: {str(e)}")
            
            self.schedule_message(1, group_tracks_task)
            
            return {
                "action": "group_tracks",
                "tracks": [track.name for track in tracks_to_group],
                "group_name": group_name,
                "track_count": len(tracks_to_group),
                "status": "grouping"
            }
            
        except Exception as e:
            return {"error": f"Failed to group tracks: {str(e)}"}

    # ========== ENHANCED TRACK CONTROLS ==========

    def handle_arm_track(self, command):
        """Arm a track for recording"""
        try:
            track_name = command.get("track", "selected")
            track = self.find_track_improved(track_name)
            
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            if hasattr(track, "arm"):
                track.arm = True
                return {"action": "arm_track", "track": track.name, "status": "armed"}
            else:
                return {"error": f"Track {track.name} cannot be armed (not an audio/MIDI track)"}
                
        except Exception as e:
            return {"error": f"Failed to arm track: {str(e)}"}

    def handle_disarm_track(self, command):
        """Disarm a track"""
        try:
            track_name = command.get("track", "selected")
            track = self.find_track_improved(track_name)
            
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            if hasattr(track, "arm"):
                track.arm = False
                return {"action": "disarm_track", "track": track.name, "status": "disarmed"}
            else:
                return {"error": f"Track {track.name} cannot be disarmed"}
                
        except Exception as e:
            return {"error": f"Failed to disarm track: {str(e)}"}

    def handle_set_track_monitor(self, command):
        """Set track monitoring mode"""
        try:
            track_name = command.get("track", "selected")
            mode = command.get("mode", "auto").lower()  # off, auto, in
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            if hasattr(track, "current_monitoring_state"):
                mode_map = {"off": 0, "auto": 1, "in": 2}
                if mode in mode_map:
                    track.current_monitoring_state = mode_map[mode]
                    return {"action": "set_track_monitor", "track": track.name, "mode": mode}
                else:
                    return {"error": f"Invalid monitor mode: {mode}"}
            else:
                return {"error": f"Track {track.name} does not support monitoring"}
                
        except Exception as e:
            return {"error": f"Failed to set track monitor: {str(e)}"}

    def handle_set_track_sends(self, command):
        """Set track send levels"""
        try:
            track_name = command.get("track", "selected")
            send_index = command.get("send", 0)  # A=0, B=1, C=2, etc.
            level = float(command.get("level", 0.0))
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            sends = track.mixer_device.sends
            if 0 <= send_index < len(sends):
                sends[send_index].value = max(0.0, min(1.0, level))
                send_letter = chr(ord('A') + send_index)
                return {
                    "action": "set_track_sends",
                    "track": track.name,
                    "send": send_letter,
                    "level": level
                }
            else:
                return {"error": f"Invalid send index: {send_index}"}
                
        except Exception as e:
            return {"error": f"Failed to set track sends: {str(e)}"}

    # ========== CLIP CONTROLS ==========

    def handle_launch_clip(self, command):
        """Launch a specific clip"""
        try:
            track_name = command.get("track", "selected")
            clip_index = command.get("clip", 0)
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            clip_slots = track.clip_slots
            if 0 <= clip_index < len(clip_slots):
                clip_slot = clip_slots[clip_index]
                if clip_slot.has_clip:
                    clip_slot.fire()
                    return {
                        "action": "launch_clip",
                        "track": track.name,
                        "clip_index": clip_index,
                        "status": "launched"
                    }
                else:
                    return {"error": f"No clip in slot {clip_index} on track {track.name}"}
            else:
                return {"error": f"Invalid clip index: {clip_index}"}
                
        except Exception as e:
            return {"error": f"Failed to launch clip: {str(e)}"}

    def handle_stop_clip(self, command):
        """Stop a specific clip"""
        try:
            track_name = command.get("track", "selected")
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}
            
            track.stop_all_clips()
            return {"action": "stop_clip", "track": track.name, "status": "stopped"}
                
        except Exception as e:
            return {"error": f"Failed to stop clip: {str(e)}"}

    def handle_launch_scene(self, command):
        """Launch a scene"""
        try:
            scene_index = command.get("scene", 0)
            scenes = self.song.scenes
            
            if 0 <= scene_index < len(scenes):
                scenes[scene_index].fire()
                return {"action": "launch_scene", "scene_index": scene_index, "status": "launched"}
            else:
                return {"error": f"Invalid scene index: {scene_index}"}
                
        except Exception as e:
            return {"error": f"Failed to launch scene: {str(e)}"}

    def handle_stop_all_clips(self, command):
        """Stop all clips"""
        try:
            self.song.stop_all_clips()
            return {"action": "stop_all_clips", "status": "all_clips_stopped"}
        except Exception as e:
            return {"error": f"Failed to stop all clips: {str(e)}"}

    # ========== ENHANCED MIXER CONTROLS ==========

    def handle_mixer_parameter(self, parameter, target, value):
        """Enhanced mixer parameters with fixed volume conversion"""
        try:
            track = self.find_track_improved(target)
            if not track:
                return {"error": f"Track not found: {target}"}

            if parameter == "mixer_volume":
                # Get current volume for relative adjustments
                current_linear = track.mixer_device.volume.value
                current_db = self._linear_to_db(current_linear)
                
                # Handle different value formats
                if isinstance(value, str):
                    if value.endswith('%'):
                        # Percentage: "50%" -> 0.5
                        percent_value = float(value.rstrip('%'))
                        live_value = percent_value / 100.0
                        actual_db = self._linear_to_db(live_value)
                    elif 'increase_' in value.lower():
                        # Relative increase: "increase_5db" or "increase_10%"
                        if 'db' in value.lower():
                            db_change = float(value.lower().replace('increase_', '').replace('db', ''))
                            new_db = current_db + db_change
                            live_value = self._db_to_linear(new_db)
                            actual_db = new_db
                        elif '%' in value:
                            percent_change = float(value.lower().replace('increase_', '').replace('%', ''))
                            new_linear = current_linear * (1 + percent_change / 100.0)
                            live_value = new_linear
                            actual_db = self._linear_to_db(live_value)
                    elif 'decrease_' in value.lower():
                        # Relative decrease: "decrease_5db" or "decrease_20%"
                        if 'db' in value.lower():
                            db_change = float(value.lower().replace('decrease_', '').replace('db', ''))
                            new_db = current_db - db_change
                            live_value = self._db_to_linear(new_db)
                            actual_db = new_db
                        elif '%' in value:
                            percent_change = float(value.lower().replace('decrease_', '').replace('%', ''))
                            new_linear = current_linear * (1 - percent_change / 100.0)
                            live_value = new_linear
                            actual_db = self._linear_to_db(live_value)
                    elif value.lower().endswith('db'):
                        # dB value: "-23db" -> -23
                        db_value = float(value.lower().rstrip('db'))
                        live_value = self._db_to_linear(db_value)
                        actual_db = db_value
                    else:
                        # Plain number, assume dB
                        db_value = float(value)
                        live_value = self._db_to_linear(db_value)
                        actual_db = db_value
                else:
                    # Numeric value, determine if it's dB or percentage
                    if value <= 1.0 and value >= 0.0:
                        # Assume linear/percentage if between 0-1
                        live_value = float(value)
                        actual_db = self._linear_to_db(live_value)
                    else:
                        # Assume dB for other values
                        db_value = float(value)
                        live_value = self._db_to_linear(db_value)
                        actual_db = db_value
                
                # Clamp to valid range
                live_value = max(0.0, min(1.0, live_value))
                
                track.mixer_device.volume.value = live_value
                return {"set": "volume", "value": f"{actual_db:.1f} dB", "track": track.name}

            elif parameter == "mixer_pan":
                live_value = max(-1.0, min(1.0, float(value) / 100.0))
                track.mixer_device.panning.value = live_value
                return {"set": "pan", "value": f"{value}%", "track": track.name}

            elif parameter == "mixer_solo":
                track.solo = bool(value)
                return {"set": "solo", "value": bool(value), "track": track.name}

            elif parameter == "mixer_mute":
                track.mute = bool(value)
                return {"set": "mute", "value": bool(value), "track": track.name}

            elif parameter == "mixer_arm":
                if hasattr(track, "arm"):
                    track.arm = bool(value)
                    return {"set": "arm", "value": bool(value), "track": track.name}
                else:
                    return {"error": f"Track {track.name} cannot be armed"}

            elif parameter.startswith("mixer_send"):
                # Handle send A, B, C etc.
                send_letter = parameter.split("_")[-1].upper()
                send_index = ord(send_letter) - ord('A')
                
                sends = track.mixer_device.sends
                if 0 <= send_index < len(sends):
                    level = max(0.0, min(1.0, float(value) / 100.0))
                    sends[send_index].value = level
                    return {"set": f"send_{send_letter}", "value": f"{value}%", "track": track.name}
                else:
                    return {"error": f"Invalid send: {send_letter}"}

            else:
                return {"error": f"Unknown mixer parameter: {parameter}"}

        except Exception as e:
            return {"error": f"Failed to set mixer parameter: {str(e)}"}

    # ========== ENHANCED DEVICE SEARCH ==========

    def search_device(self, query):
        """Enhanced device search with better matching"""
        query = str(query).lower().strip()

        # Exact match first
        if query in self.device_cache:
            return self.device_cache[query]

        # Enhanced priority mapping including missing devices
        priority_map = {
            'reverb': ['reverb'],
            'delay': ['delay', 'echo'],
            'compressor': ['compressor'],
            'comp': ['compressor'],
            'eq': ['eq eight', 'eq three'],
            'equalizer': ['eq eight'],
            'saturator': ['saturator'],
            'distortion': ['saturator', 'overdrive'],
            'filter': ['auto filter', 'filter'],
            'gate': ['gate'],
            'limiter': ['limiter'],
            'drum bus': ['drum bus'],  # Fixed mapping for "Drum Buss"
            'drum buss': ['drum bus'], # Alternative spelling
            'glue comp': ['glue compressor'],
            'glue compressor': ['glue compressor'],
            'multiband': ['multiband dynamics'],
            'chorus': ['chorus'],
            'flanger': ['flanger'],
            'phaser': ['phaser']
        }

        if query in priority_map:
            for priority_term in priority_map[query]:
                for key, item_data in self.device_cache.items():
                    if priority_term in key:
                        return item_data

        # Partial matches as fallback
        matches = []
        for key, item_data in self.device_cache.items():
            if query in key:
                matches.append((len(key), key, item_data))

        matches.sort()
        return matches[0][2] if matches else None

    # ========== IMPROVED TRACK FINDING ==========

    def find_track_improved(self, track_identifier):
        """Enhanced track finding with better matching"""
        try:
            tracks = list(self.song.tracks)

            # Handle "selected" case
            if str(track_identifier).lower() == "selected":
                try:
                    selected_track = self.song.view.selected_track
                    if selected_track in tracks:
                        return selected_track
                    return tracks[0] if len(tracks) > 0 else None
                except:
                    return tracks[0] if len(tracks) > 0 else None

            # Handle integer index
            if isinstance(track_identifier, int):
                if 0 <= track_identifier < len(tracks):
                    return tracks[track_identifier]
                return None

            # Handle string name with improved matching
            track_name = str(track_identifier).lower().strip()
            
            # Direct exact match first
            for track in tracks:
                if track.name.lower() == track_name:
                    return track
            
            # Then try partial matches
            for track in tracks:
                if track_name in track.name.lower() or track.name.lower() in track_name:
                    return track

            return None

        except Exception as e:
            self.log_message(f"Error finding track '{track_identifier}': {str(e)}")
            return None

    # ========== ENHANCED SESSION INFO ==========

    def handle_get_detailed_session_info(self, command):
        """Enhanced session information"""
        try:
            selected_track_name = self._get_selected_track_name()
            
            track_details = []
            tracks = list(self.song.tracks)
            
            for i in range(len(tracks)):
                track = tracks[i]
                
                volume_db = self._linear_to_db(track.mixer_device.volume.value)
                pan_percent = int(track.mixer_device.panning.value * 100)
                
                # Get send levels
                sends_info = []
                for j, send in enumerate(track.mixer_device.sends):
                    send_letter = chr(ord('A') + j)
                    sends_info.append({
                        "send": send_letter,
                        "level": round(send.value * 100, 1)
                    })
                
                effects_info = []
                for device in track.devices:
                    effect_info = {
                        "name": device.name,
                        "active": device.is_active,
                        "type": device.class_name
                    }
                    effects_info.append(effect_info)
                
                track_info = {
                    "index": i,
                    "name": track.name,
                    "type": "audio" if track.has_audio_input else "midi",
                    "selected": track.name == selected_track_name,
                    "volume_db": round(volume_db, 1),
                    "pan_percent": pan_percent,
                    "muted": track.mute,
                    "soloed": track.solo,
                    "armed": getattr(track, "arm", False),
                    "sends": sends_info,
                    "effects": effects_info
                }
                track_details.append(track_info)
            
            comprehensive_state = {
                "tempo": self.song.tempo,
                "time_signature": [self.song.signature_numerator, self.song.signature_denominator],
                "is_playing": self.song.is_playing,
                "is_recording": getattr(self.song, "record_mode", False),
                "loop_enabled": self.song.loop,
                "metronome_enabled": self.song.metronome,
                "selected_track": selected_track_name,
                "tracks": track_details,
                "track_count": len(tracks),
                "device_cache_info": {
                    "total_devices": len(self.device_cache),
                    "audio_effects": len([item for item in self.device_cache.values() if item["category"] == "audio_effects"]),
                    "instruments": len([item for item in self.device_cache.values() if item["category"] == "instruments"]),
                    "midi_effects": len([item for item in self.device_cache.values() if item["category"] == "midi_effects"])
                }
            }
            
            return comprehensive_state
            
        except Exception as e:
            return {"error": f"Failed to get detailed session info: {str(e)}"}

    def _linear_to_db(self, linear_value):
        """Convert linear volume (0-1) to dB"""
        if linear_value <= 0:
            return -70.0
        return 20 * math.log10(linear_value / 0.85)

    def _db_to_linear(self, db_value):
        """Convert dB to linear volume (0-1)"""
        if db_value <= -70:
            return 0.0
        elif db_value >= 6:
            return 1.0
        else:
            # Convert dB to linear using Ableton's reference level
            return 0.85 * pow(10, db_value / 20.0)

    def _get_selected_track_name(self):
        """Get the name of currently selected track"""
        try:
            selected_track = self.song.view.selected_track
            return selected_track.name if selected_track else "None"
        except:
            return "None"

    # ========== SOCKET SERVER AND REMAINING HANDLERS ==========

    def start_socket_server(self):
        """Start the socket server for MCP communication"""
        try:
            self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_server.bind((self.socket_host, self.socket_port))
            self.socket_server.listen(1)

            self.is_running = True
            self.server_thread = threading.Thread(target=self.socket_server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            self.log_message(f"Socket server started on {self.socket_host}:{self.socket_port}")

        except Exception as e:
            self.log_message(f"Failed to start socket server: {str(e)}")

    def socket_server_loop(self):
        """Main socket server loop"""
        while self.is_running:
            try:
                self.client_socket, client_address = self.socket_server.accept()
                self.log_message(f"MCP client connected: {client_address}")

                while self.is_running:
                    try:
                        data = self.client_socket.recv(4096)
                        if not data:
                            break

                        command_str = data.decode("utf-8")
                        response = self.process_command(command_str)

                        response_str = json.dumps(response)
                        self.client_socket.send(response_str.encode("utf-8"))

                    except Exception as e:
                        self.log_message(f"Error processing command: {str(e)}")
                        break

                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None

            except Exception as e:
                if self.is_running:
                    self.log_message(f"Socket server error: {str(e)}")
                time.sleep(1)

    def process_command(self, command_str):
        """Process incoming command from MCP server with enhanced debugging"""
        try:
            self.log_message(f"Received raw command: {command_str}")
            command = json.loads(command_str)
            action = command.get("action")
            
            self.log_message(f"Parsed command - Action: {action}, Full command: {command}")

            if action in self.command_handlers:
                self.log_message(f"Found handler for action: {action}")
                result = self.command_handlers[action](command)
                self.log_message(f"Handler completed for {action}, result: {result}")
                return {"status": "success", "result": result, "action": action}
            else:
                error_msg = f"Unknown action: {action}"
                self.log_message(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "action": action,
                }

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON command: {str(e)}"
            self.log_message(error_msg)
            return {"status": "error", "message": "Invalid JSON command"}
        except Exception as e:
            error_msg = f"Unexpected error in process_command: {str(e)}"
            self.log_message(error_msg)
            return {"status": "error", "message": str(e)}

    def update_track_cache(self):
        """Update the track name to index mapping"""
        try:
            self.track_name_cache = {}
            tracks = list(self.song.tracks)

            for i in range(len(tracks)):
                track = tracks[i]
                track_name = str(track.name).lower()
                self.track_name_cache[track_name] = i

            self.log_message(f"Track cache updated: {len(self.track_name_cache)} tracks")

        except Exception as e:
            self.log_message(f"Error updating track cache: {str(e)}")

    # ========== REMAINING HANDLERS ==========

    def handle_create_tracks(self, command):
        """Handle track creation"""
        try:
            track_type = command.get("track_type", "midi").lower()
            count = command.get("count", 1)
            names = command.get("names", [])

            def create_tracks_task():
                try:
                    for i in range(count):
                        if track_type == "midi":
                            self.song.create_midi_track(-1)
                        elif track_type == "audio":
                            self.song.create_audio_track(-1)

                        tracks = list(self.song.tracks)
                        new_track = tracks[len(tracks) - 1]

                        if i < len(names):
                            new_track.name = names[i]

                    self.update_track_cache()
                    self.log_message(f"Created {count} {track_type} tracks")

                except Exception as e:
                    self.log_message(f"Error creating tracks: {str(e)}")

            self.schedule_message(1, create_tracks_task)

            return {
                "status": "scheduled",
                "message": f"Creating {count} {track_type} tracks",
                "count": count,
                "type": track_type,
            }

        except Exception as e:
            return {"error": f"Failed to create tracks: {str(e)}"}

    def handle_add_audio_effect(self, command):
        """Add audio effect with improved track targeting"""
        try:
            track_name = command.get("track", "selected")
            effect_name = command.get("effect", "").lower()
            
            target_track = self.find_track_improved(track_name)
            if not target_track:
                return {"error": f"Track not found: {track_name}"}

            device_data = self.search_device(effect_name)
            if not device_data:
                return {"error": f"Device '{effect_name}' not found"}

            def load_device_task():
                try:
                    result = self.load_device_to_track(effect_name, target_track)
                    self.log_message(f"Device loading result: {result}")
                except Exception as e:
                    self.log_message(f"Error in load_device_task: {str(e)}")

            self.schedule_message(1, load_device_task)

            return {
                "action": "add_audio_effect",
                "track": target_track.name,
                "effect": effect_name,
                "device_name": device_data["name"],
                "device_category": device_data["category"],
                "status": "loading_device"
            }

        except Exception as e:
            return {"error": f"Failed to add audio effect: {str(e)}"}

    def load_device_to_track(self, device_name, track):
        """Load device onto specific track"""
        try:
            self.log_message(f"Loading device: {device_name} onto track: {track.name}")

            device_data = self.search_device(device_name)

            if not device_data:
                self.log_message(f"Device '{device_name}' not found in cache")
                return f"ERROR: Device '{device_name}' not found"

            self.song.view.selected_track = track
            time.sleep(0.1)

            browser_item = device_data["item"]
            self.browser.load_item(browser_item)

            success_msg = f"SUCCESS: Loaded '{device_data['name']}' onto '{track.name}'"
            self.log_message(success_msg)
            return success_msg

        except Exception as e:
            error_msg = f"ERROR loading device: {str(e)}"
            self.log_message(error_msg)
            return error_msg

    def handle_transport_play(self, command):
        """Handle play command"""
        try:
            self.song.start_playing()
            return {"action": "play", "status": "started"}
        except Exception as e:
            return {"error": f"Failed to start playback: {str(e)}"}

    def handle_transport_stop(self, command):
        """Handle stop command"""
        try:
            self.song.stop_playing()
            return {"action": "stop", "status": "stopped"}
        except Exception as e:
            return {"error": f"Failed to stop playback: {str(e)}"}

    def handle_transport_record(self, command):
        """Handle record command"""
        try:
            self.song.record_mode = not self.song.record_mode
            return {"action": "record", "enabled": self.song.record_mode}
        except Exception as e:
            return {"error": f"Failed to toggle recording: {str(e)}"}

    # Additional handlers for remaining functionality...
    def handle_remove_audio_effect(self, command):
        """Remove audio effect"""
        try:
            track_name = command.get("track")
            effect_index = command.get("effect_index", -1)
            
            track = self.find_track_improved(track_name)
            if not track:
                return {"error": f"Track not found: {track_name}"}

            devices = list(track.devices)
            if not devices:
                return {"error": f"No effects on track {track_name}"}

            if effect_index == -1:
                effect_index = len(devices) - 1
            
            if 0 <= effect_index < len(devices):
                device_name = devices[effect_index].name
                track.delete_device(effect_index)
                return {
                    "action": "remove_audio_effect",
                    "track": track.name,
                    "removed_effect": device_name,
                    "index": effect_index
                }
            else:
                return {"error": f"Invalid effect index: {effect_index}"}

        except Exception as e:
            return {"error": f"Failed to remove audio effect: {str(e)}"}

    def handle_get_track_effects(self, command):
        """Get all effects on a track"""
        try:
            track_name = command.get("track")
            track = self.find_track_improved(track_name)
            
            if not track:
                return {"error": f"Track not found: {track_name}"}

            effects_info = []
            for i, device in enumerate(track.devices):
                effect_info = {
                    "index": i,
                    "name": device.name,
                    "type": device.class_name,
                    "is_active": device.is_active
                }
                effects_info.append(effect_info)

            return {
                "track": track.name,
                "effects": effects_info,
                "count": len(effects_info)
            }

        except Exception as e:
            return {"error": f"Failed to get track effects: {str(e)}"}

    def handle_get_session_info(self, command):
        """Get basic session information"""
        try:
            track_info = []
            tracks = list(self.song.tracks)
            for i in range(len(tracks)):
                track = tracks[i]
                effects = [device.name for device in track.devices]
                track_info.append({
                    "index": i,
                    "name": track.name,
                    "type": "audio" if track.has_audio_input else "midi",
                    "muted": track.mute,
                    "soloed": track.solo,
                    "armed": getattr(track, "arm", False),
                    "effects": effects,
                    "effect_count": len(effects)
                })
            return {
                "tempo": self.song.tempo,
                "is_playing": self.song.is_playing,
                "track_count": len(tracks),
                "tracks": track_info,
            }
        except Exception as e:
            return {"error": f"Failed to get session info: {str(e)}"}

    def handle_get_available_devices(self, command):
        """Get list of available devices"""
        try:
            category = command.get("category", "all").lower()
            limit = command.get("limit", 20)
            
            if category == "all":
                result = {}
                categories = set(item["category"] for item in self.device_cache.values())
                
                for cat in categories:
                    cat_devices = [item for item in self.device_cache.values() if item["category"] == cat]
                    result[cat] = [{"name": item["name"]} for item in cat_devices[:10]]
                
                return {
                    "available_devices": result, 
                    "total_devices": len(self.device_cache),
                    "type": "enhanced_devices"
                }
            else:
                devices = [item for item in self.device_cache.values() if category in item["category"]]
                device_list = [{"name": item["name"]} for item in devices[:limit]]
                
                return {"category": category, "devices": device_list, "count": len(devices)}
                
        except Exception as e:
            return {"error": f"Failed to get available devices: {str(e)}"}

    def handle_search_devices(self, command):
        """Search for devices matching a query"""
        try:
            query = command.get("query", "")
            limit = command.get("limit", 10)
            
            if not query:
                return {"error": "No search query provided"}
            
            device_data = self.search_device(query)
            if device_data:
                return {
                    "query": query,
                    "matches": [{
                        "name": device_data["name"],
                        "category": device_data["category"]
                    }],
                    "count": 1
                }
            else:
                return {
                    "query": query,
                    "matches": [],
                    "count": 0
                }
            
        except Exception as e:
            return {"error": f"Failed to search devices: {str(e)}"}

    def handle_browse_devices(self, command):
        """Browse devices by category"""
        try:
            category = command.get("category", "audio_effects")
            return self.handle_get_available_devices({"category": category})
        except Exception as e:
            return {"error": f"Failed to browse devices: {str(e)}"}

    def handle_load_preset(self, command):
        """Load a preset"""
        try:
            preset_name = command.get("preset", "")
            track_name = command.get("track", "selected")
            
            # This would need to be implemented based on Live's preset system
            return {"error": "Preset loading not yet implemented"}
        except Exception as e:
            return {"error": f"Failed to load preset: {str(e)}"}

    def log_message(self, message):
        """Log message to Live's log"""
        try:
            self.c_instance.log_message(f"Enhanced MCP: {message}")
        except:
            pass


# Required function for Ableton to load the Remote Script
def create_instance(c_instance):
    """Create and return the Remote Script instance"""
    return EnhancedAbletonMCP(c_instance)