# =========================================== BOT CODE STARTS HERE ===========================================
import discord
from discord.ext import commands, tasks
from discord import ui, app_commands
import os
import random
import string
import json
import subprocess
from dotenv import load_dotenv
import asyncio
import datetime
import docker
import time
import logging
import traceback
import aiohttp
import socket
import re
import psutil
import platform
import shutil
from typing import Optional, Literal
import sqlite3
import pickle
import base64
import threading
import textwrap
import math
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('VantaNode_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('[VantaNode]')

# Load environment variables
load_dotenv()

# Bot configuration
TOKEN = os.getenv('DISCORD_TOKEN')
ADMIN_IDS = {int(id_) for id_ in os.getenv('ADMIN_IDS', '1210291131301101618').split(',') if id_.strip()}
ADMIN_ROLE_ID = int(os.getenv('ADMIN_ROLE_ID', '1376177459870961694'))
WATERMARK = "VantaNode VPS Service"
WELCOME_MESSAGE = "Welcome To VantaNode! Get Started With Us!"
MAX_VPS_PER_USER = int(os.getenv('MAX_VPS_PER_USER', '3'))
DEFAULT_OS_IMAGE = os.getenv('DEFAULT_OS_IMAGE', 'ubuntu:22.04')
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'bridge')
MAX_CONTAINERS = int(os.getenv('MAX_CONTAINERS', '100'))
DB_FILE = 'VantaNode.db'
BACKUP_FILE = 'VantaNode_backup.pkl'

# Resource tracking
CPU_CORES_AVAILABLE = psutil.cpu_count(logical=False)
ALLOCATED_CPU_CORES = set()
VPS_UPTIME_TRACKER = {}

# Dockerfile template for custom images - OPTIMIZED FOR SPEED
DOCKERFILE_TEMPLATE = """FROM {base_image}

# Prevent prompts and use faster mirrors
ENV DEBIAN_FONTEND=noninteractive

# Use faster APT sources and install minimal packages
RUN sed -i 's/archive.ubuntu.com/mirror.rackspace.com/g' /etc/apt/sources.list && \\
    apt-get update && \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
        sudo openssh-server tmate neofetch htop nano wget curl git tmux && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup user
RUN useradd -m -s /bin/bash {username} && \\
    echo "{username}:{user_password}" | chpasswd && \\
    echo "root:{root_password}" | chpasswd && \\
    usermod -aG sudo {username}

# SSH configuration
RUN mkdir -p /var/run/sshd && \\
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \\
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Customization
RUN echo '{welcome_message}' > /etc/motd && \\
    echo 'echo "{welcome_message}"' >> /home/{username}/.bashrc && \\
    echo '{watermark}' > /etc/machine-info && \\
    echo 'vantanode-{vps_id}' > /etc/hostname

CMD ["/usr/sbin/sshd", "-D"]
"""

class Database:
    """Handles all data persistence using SQLite3"""
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._initialize_settings()

    def _create_tables(self):
        """Create necessary tables"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vps_instances (
                token TEXT PRIMARY KEY,
                vps_id TEXT UNIQUE,
                container_id TEXT,
                memory INTEGER,
                cpu INTEGER,
                disk INTEGER,
                username TEXT,
                password TEXT,
                root_password TEXT,
                created_by TEXT,
                created_at TEXT,
                tmate_session TEXT,
                watermark TEXT,
                os_image TEXT,
                restart_count INTEGER DEFAULT 0,
                last_restart TEXT,
                status TEXT DEFAULT 'running',
                use_custom_image BOOLEAN DEFAULT 1,
                allocated_cpus TEXT,
                uptime_start TEXT,
                total_uptime INTEGER DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                key TEXT PRIMARY KEY,
                value INTEGER DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS banned_users (
                user_id TEXT PRIMARY KEY
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                user_id TEXT PRIMARY KEY
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_usage (
                vps_id TEXT PRIMARY KEY,
                cpu_usage REAL DEFAULT 0,
                memory_usage REAL DEFAULT 0,
                disk_usage REAL DEFAULT 0,
                network_tx REAL DEFAULT 0,
                network_rx REAL DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        self.conn.commit()

    def _initialize_settings(self):
        """Initialize default settings"""
        defaults = {
            'max_containers': str(MAX_CONTAINERS),
            'max_vps_per_user': str(MAX_VPS_PER_USER),
            'total_allocated_memory': '0',
            'total_allocated_cpu': '0',
            'total_allocated_disk': '0'
        }
        for key, value in defaults.items():
            self.cursor.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', (key, value))
        
        # Load admin users from database
        self.cursor.execute('SELECT user_id FROM admin_users')
        for row in self.cursor.fetchall():
            ADMIN_IDS.add(int(row[0]))
        
        # Initialize uptime for running VPS
        self.cursor.execute("SELECT vps_id, container_id FROM vps_instances WHERE status = 'running'")
        for vps_id, container_id in self.cursor.fetchall():
            VPS_UPTIME_TRACKER[vps_id] = {
                'start_time': datetime.datetime.now(),
                'container_id': container_id
            }
            
        self.conn.commit()

    def get_setting(self, key, default=None):
        self.cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return int(result[0]) if result else default

    def set_setting(self, key, value):
        self.cursor.execute('INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)', (key, str(value)))
        self.conn.commit()

    def update_resource_usage(self, vps_id, cpu, memory, disk, net_tx, net_rx):
        """Update resource usage statistics"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO resource_usage 
            (vps_id, cpu_usage, memory_usage, disk_usage, network_tx, network_rx, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (vps_id, cpu, memory, disk, net_tx, net_rx, datetime.datetime.now().isoformat()))
        self.conn.commit()

    def get_resource_usage(self, vps_id):
        """Get resource usage for a VPS"""
        self.cursor.execute('SELECT * FROM resource_usage WHERE vps_id = ?', (vps_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def update_uptime(self, vps_id, uptime_seconds):
        """Update total uptime for VPS"""
        self.cursor.execute('''
            UPDATE vps_instances 
            SET total_uptime = total_uptime + ?
            WHERE vps_id = ?
        ''', (uptime_seconds, vps_id))
        self.conn.commit()

    def get_stat(self, key, default=0):
        self.cursor.execute('SELECT value FROM usage_stats WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else default

    def increment_stat(self, key, amount=1):
        current = self.get_stat(key)
        self.cursor.execute('INSERT OR REPLACE INTO usage_stats (key, value) VALUES (?, ?)', (key, current + amount))
        self.conn.commit()

    def get_vps_by_id(self, vps_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE vps_id = ?', (vps_id,))
        row = self.cursor.fetchone()
        if not row:
            return None, None
        columns = [desc[0] for desc in self.cursor.description]
        vps = dict(zip(columns, row))
        return vps['token'], vps

    def get_vps_by_token(self, token):
        self.cursor.execute('SELECT * FROM vps_instances WHERE token = ?', (token,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_user_vps_count(self, user_id):
        self.cursor.execute('SELECT COUNT(*) FROM vps_instances WHERE created_by = ?', (str(user_id),))
        return self.cursor.fetchone()[0]

    def get_user_vps(self, user_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE created_by = ?', (str(user_id),))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def get_all_vps(self):
        self.cursor.execute('SELECT * FROM vps_instances')
        columns = [desc[0] for desc in self.cursor.description]
        return {row[0]: dict(zip(columns, row)) for row in self.cursor.fetchall()}

    def add_vps(self, vps_data):
        columns = ', '.join(vps_data.keys())
        placeholders = ', '.join('?' for _ in vps_data)
        self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps_data.values()))
        
        # Update total allocated resources
        memory = vps_data.get('memory', 0)
        cpu = vps_data.get('cpu', 0)
        disk = vps_data.get('disk', 0)
        
        total_mem = self.get_setting('total_allocated_memory', 0) + memory
        total_cpu = self.get_setting('total_allocated_cpu', 0) + cpu
        total_disk = self.get_setting('total_allocated_disk', 0) + disk
        
        self.set_setting('total_allocated_memory', total_mem)
        self.set_setting('total_allocated_cpu', total_cpu)
        self.set_setting('total_allocated_disk', total_disk)
        
        self.conn.commit()
        self.increment_stat('total_vps_created')

    def remove_vps(self, token):
        self.cursor.execute('SELECT memory, cpu, disk FROM vps_instances WHERE token = ?', (token,))
        row = self.cursor.fetchone()
        if row:
            memory, cpu, disk = row
            
            # Update total allocated resources
            total_mem = self.get_setting('total_allocated_memory', 0) - memory
            total_cpu = self.get_setting('total_allocated_cpu', 0) - cpu
            total_disk = self.get_setting('total_allocated_disk', 0) - disk
            
            self.set_setting('total_allocated_memory', max(0, total_mem))
            self.set_setting('total_allocated_cpu', max(0, total_cpu))
            self.set_setting('total_allocated_disk', max(0, total_disk))
        
        self.cursor.execute('DELETE FROM vps_instances WHERE token = ?', (token,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def update_vps(self, token, updates):
        set_clause = ', '.join(f'{k} = ?' for k in updates)
        values = list(updates.values()) + [token]
        self.cursor.execute(f'UPDATE vps_instances SET {set_clause} WHERE token = ?', values)
        self.conn.commit()
        return self.cursor.rowcount > 0

    def is_user_banned(self, user_id):
        self.cursor.execute('SELECT 1 FROM banned_users WHERE user_id = ?', (str(user_id),))
        return self.cursor.fetchone() is not None

    def ban_user(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO banned_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()

    def unban_user(self, user_id):
        self.cursor.execute('DELETE FROM banned_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()

    def get_banned_users(self):
        self.cursor.execute('SELECT user_id FROM banned_users')
        return [row[0] for row in self.cursor.fetchall()]

    def add_admin(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO admin_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()
        ADMIN_IDS.add(int(user_id))

    def remove_admin(self, user_id):
        self.cursor.execute('DELETE FROM admin_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()
        if int(user_id) in ADMIN_IDS:
            ADMIN_IDS.remove(int(user_id))

    def get_admins(self):
        self.cursor.execute('SELECT user_id FROM admin_users')
        return [row[0] for row in self.cursor.fetchall()]

    def backup_data(self):
        """Backup all data to a file"""
        data = {
            'vps_instances': self.get_all_vps(),
            'usage_stats': {},
            'system_settings': {},
            'banned_users': self.get_banned_users(),
            'admin_users': self.get_admins()
        }
        
        # Get usage stats
        self.cursor.execute('SELECT * FROM usage_stats')
        for row in self.cursor.fetchall():
            data['usage_stats'][row[0]] = row[1]
            
        # Get system settings
        self.cursor.execute('SELECT * FROM system_settings')
        for row in self.cursor.fetchall():
            data['system_settings'][row[0]] = row[1]
            
        with open(BACKUP_FILE, 'wb') as f:
            pickle.dump(data, f)
            
        return True

    def restore_data(self):
        """Restore data from backup file"""
        if not os.path.exists(BACKUP_FILE):
            return False
            
        try:
            with open(BACKUP_FILE, 'rb') as f:
                data = pickle.load(f)
                
            # Clear all tables
            self.cursor.execute('DELETE FROM vps_instances')
            self.cursor.execute('DELETE FROM usage_stats')
            self.cursor.execute('DELETE FROM system_settings')
            self.cursor.execute('DELETE FROM banned_users')
            self.cursor.execute('DELETE FROM admin_users')
            
            # Restore VPS instances
            for token, vps in data['vps_instances'].items():
                columns = ', '.join(vps.keys())
                placeholders = ', '.join('?' for _ in vps)
                self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps.values()))
            
            # Restore usage stats
            for key, value in data['usage_stats'].items():
                self.cursor.execute('INSERT INTO usage_stats (key, value) VALUES (?, ?)', (key, value))
                
            # Restore system settings
            for key, value in data['system_settings'].items():
                self.cursor.execute('INSERT INTO system_settings (key, value) VALUES (?, ?)', (key, value))
                
            # Restore banned users
            for user_id in data['banned_users']:
                self.cursor.execute('INSERT INTO banned_users (user_id) VALUES (?)', (user_id,))
                
            # Restore admin users
            for user_id in data['admin_users']:
                self.cursor.execute('INSERT INTO admin_users (user_id) VALUES (?)', (user_id,))
                ADMIN_IDS.add(int(user_id))
                
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False

    def close(self):
        self.conn.close()

# Initialize bot with command prefix '/'
class VantaNodeBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = Database(DB_FILE)
        self.session = None
        self.docker_client = None
        self.system_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_io': (0, 0),
            'last_updated': 0
        }
        self.my_persistent_views = {}

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
            self.loop.create_task(self.update_system_stats())
            self.loop.create_task(self.update_resource_usage())
            await self.reconnect_containers()
            await self.restore_persistent_views()
            
            # Start uptime monitor after bot is ready
            self.uptime_monitor.start()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

    async def reconnect_containers(self):
        """Reconnect to existing containers on startup"""
        if not self.docker_client:
            return
            
        for token, vps in list(self.db.get_all_vps().items()):
            if vps['status'] == 'running':
                try:
                    container = self.docker_client.containers.get(vps['container_id'])
                    if container.status != 'running':
                        container.start()
                    # Update uptime tracker
                    VPS_UPTIME_TRACKER[vps['vps_id']] = {
                        'start_time': datetime.datetime.now(),
                        'container_id': vps['container_id']
                    }
                    logger.info(f"Reconnected and started container for VPS {vps['vps_id']}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {vps['container_id']} not found, removing from data")
                    self.db.remove_vps(token)
                except Exception as e:
                    logger.error(f"Error reconnecting container {vps['vps_id']}: {e}")

    async def restore_persistent_views(self):
        """Restore persistent views after restart"""
        pass

    async def update_system_stats(self):
        """Update system statistics periodically"""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                mem = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # Network IO
                net_io = psutil.net_io_counters()
                
                self.system_stats = {
                    'cpu_usage': cpu_percent,
                    'memory_usage': mem.percent,
                    'memory_used': mem.used / (1024 ** 3),  # GB
                    'memory_total': mem.total / (1024 ** 3),  # GB
                    'disk_usage': disk.percent,
                    'disk_used': disk.used / (1024 ** 3),  # GB
                    'disk_total': disk.total / (1024 ** 3),  # GB
                    'network_sent': net_io.bytes_sent / (1024 ** 2),  # MB
                    'network_recv': net_io.bytes_recv / (1024 ** 2),  # MB
                    'last_updated': time.time()
                }
            except Exception as e:
                logger.error(f"Error updating system stats: {e}")
            await asyncio.sleep(30)

    async def update_resource_usage(self):
        """Update resource usage for all running VPS"""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                if not self.docker_client:
                    await asyncio.sleep(60)
                    continue
                    
                for token, vps in list(self.db.get_all_vps().items()):
                    if vps['status'] != 'running':
                        continue
                        
                    try:
                        container = self.docker_client.containers.get(vps['container_id'])
                        stats = container.stats(stream=False)
                        
                        # Calculate CPU usage
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                        cpu_percent = 0.0
                        if system_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * 100.0 * CPU_CORES_AVAILABLE
                        
                        # Memory usage
                        memory_usage = stats['memory_stats']['usage'] / (1024 ** 3)  # GB
                        memory_limit = vps['memory']
                        memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                        
                        # Network usage
                        networks = stats.get('networks', {})
                        rx_bytes = sum(net['rx_bytes'] for net in networks.values()) / (1024 ** 2)  # MB
                        tx_bytes = sum(net['tx_bytes'] for net in networks.values()) / (1024 ** 2)  # MB
                        
                        # Update database
                        self.db.update_resource_usage(
                            vps['vps_id'],
                            cpu_percent,
                            memory_percent,
                            0,  # Disk usage placeholder
                            tx_bytes,
                            rx_bytes
                        )
                        
                    except Exception as e:
                        logger.error(f"Error updating resource usage for {vps['vps_id']}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in resource usage update loop: {e}")
                
            await asyncio.sleep(60)  # Update every minute

    @tasks.loop(minutes=5)
    async def uptime_monitor(self):
        """Monitor and update uptime for all running VPS"""
        try:
            current_time = datetime.datetime.now()
            for vps_id, tracker in list(VPS_UPTIME_TRACKER.items()):
                try:
                    # Calculate uptime since last check
                    uptime_seconds = (current_time - tracker['start_time']).total_seconds()
                    
                    # Update database
                    self.db.update_uptime(vps_id, int(uptime_seconds))
                    
                    # Reset start time
                    tracker['start_time'] = current_time
                    
                except Exception as e:
                    logger.error(f"Error updating uptime for {vps_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in uptime monitor: {e}")

    @uptime_monitor.before_loop
    async def before_uptime_monitor(self):
        await self.wait_until_ready()

    async def close(self):
        # Update uptime before closing
        current_time = datetime.datetime.now()
        for vps_id, tracker in list(VPS_UPTIME_TRACKER.items()):
            try:
                uptime_seconds = (current_time - tracker['start_time']).total_seconds()
                self.db.update_uptime(vps_id, int(uptime_seconds))
            except Exception as e:
                logger.error(f"Error updating uptime on close for {vps_id}: {e}")
        
        await super().close()
        if self.session:
            await self.session.close()
        if self.docker_client:
            self.docker_client.close()
        self.db.close()

def generate_token():
    """Generate a random token for VPS access"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=24))

def generate_vps_id():
    """Generate a unique VPS ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def generate_ssh_password():
    """Generate a random SSH password"""
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choices(chars, k=16))

def allocate_cpu_cores(cpu_cores):
    """Allocate specific CPU cores for isolation"""
    global ALLOCATED_CPU_CORES
    
    available_cores = [c for c in range(CPU_CORES_AVAILABLE) if c not in ALLOCATED_CPU_CORES]
    
    if len(available_cores) < cpu_cores:
        raise Exception(f"Not enough CPU cores available. Requested: {cpu_cores}, Available: {len(available_cores)}")
    
    allocated = available_cores[:cpu_cores]
    ALLOCATED_CPU_CORES.update(allocated)
    
    # Convert to cpuset format (0,1,2,3)
    return ','.join(str(core) for core in allocated)

def free_cpu_cores(cpu_set):
    """Free allocated CPU cores"""
    global ALLOCATED_CPU_CORES
    if cpu_set:
        cores = [int(c) for c in cpu_set.split(',')]
        ALLOCATED_CPU_CORES.difference_update(cores)

def has_admin_role(ctx):
    """Check if user has admin role or is in ADMIN_IDS"""
    if isinstance(ctx, discord.Interaction):
        user_id = ctx.user.id
        roles = ctx.user.roles
    else:
        user_id = ctx.author.id
        roles = ctx.author.roles
    
    if user_id in ADMIN_IDS:
        return True
    
    return any(role.id == ADMIN_ROLE_ID for role in roles)

async def capture_ssh_session_line(process):
    """Capture the SSH session line from tmate output"""
    try:
        while True:
            output = await process.stdout.readline()
            if not output:
                break
            output = output.decode('utf-8').strip()
            if "ssh session:" in output:
                return output.split("ssh session:")[1].strip()
        return None
    except Exception as e:
        logger.error(f"Error capturing SSH session: {e}")
        return None

async def run_docker_command(container_id, command, timeout=30):
    """Run a Docker command asynchronously with timeout"""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            if process.returncode != 0:
                raise Exception(f"Command failed: {stderr.decode()}")
            return True, stdout.decode()
        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Command timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Error running Docker command: {e}")
        return False, str(e)

async def build_custom_image(vps_id, username, root_password, user_password, base_image=DEFAULT_OS_IMAGE):
    """Build a custom Docker image using our OPTIMIZED template"""
    try:
        # Create a temporary directory for the Dockerfile
        temp_dir = f"temp_dockerfiles/{vps_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate Dockerfile content with optimized settings
        dockerfile_content = DOCKERFILE_TEMPLATE.format(
            base_image=base_image,
            root_password=root_password,
            username=username,
            user_password=user_password,
            welcome_message=WELCOME_MESSAGE,
            watermark=WATERMARK,
            vps_id=vps_id
        )
        
        # Write Dockerfile
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Build the image with cache and parallel builds
        image_tag = f"vantanode/{vps_id.lower()}:latest"
        build_process = await asyncio.create_subprocess_exec(
            "docker", "build", "--pull", "--rm", "-t", image_tag, temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await build_process.communicate()
        
        if build_process.returncode != 0:
            raise Exception(f"Failed to build image: {stderr.decode()}")
        
        return image_tag
    except Exception as e:
        logger.error(f"Error building custom image: {e}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")

async def setup_container(container_id, status_msg, memory, username, vps_id=None, use_custom_image=False):
    """FAST container setup with VantaNode customization"""
    try:
        # Ensure container is running
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🔍 Checking container...", ephemeral=True)
        else:
            await status_msg.edit(content="🔍 Checking container...")
            
        container = bot.docker_client.containers.get(container_id)
        if container.status != "running":
            if isinstance(status_msg, discord.Interaction):
                await status_msg.followup.send("🚀 Starting container...", ephemeral=True)
            else:
                await status_msg.edit(content="🚀 Starting container...")
            container.start()
            await asyncio.sleep(2)

        # Generate SSH password
        ssh_password = generate_ssh_password()
        
        # Only install packages if not using custom image
        if not use_custom_image:
            if isinstance(status_msg, discord.Interaction):
                await status_msg.followup.send("📦 Installing packages...", ephemeral=True)
            else:
                await status_msg.edit(content="📦 Installing packages...")
                
            # Update package list with fast mirror
            success, output = await run_docker_command(container_id, [
                "bash", "-c", 
                "sed -i 's/archive.ubuntu.com/mirror.rackspace.com/g' /etc/apt/sources.list && apt-get update"
            ])
            if not success:
                logger.warning(f"Failed to update package list: {output}")

            # Install minimal required packages
            packages = ["tmate", "openssh-server", "sudo", "neofetch"]
            success, output = await run_docker_command(container_id, [
                "apt-get", "install", "-y", "--no-install-recommends"
            ] + packages)
            if not success:
                logger.warning(f"Failed to install packages: {output}")

        # Setup SSH
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🔐 Configuring SSH...", ephemeral=True)
        else:
            await status_msg.edit(content="🔐 Configuring SSH...")
            
        # Create user and set password (if not using custom image)
        if not use_custom_image:
            user_setup_commands = [
                f"useradd -m -s /bin/bash {username}",
                f"echo '{username}:{ssh_password}' | chpasswd",
                f"usermod -aG sudo {username}",
                "mkdir -p /var/run/sshd",
                "sed -i 's/#PermitRootLogin prohibit-root/PermitRootLogin yes/' /etc/ssh/sshd_config",
                "sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config",
                "/usr/sbin/sshd"
            ]
            
            for cmd in user_setup_commands:
                success, output = await run_docker_command(container_id, ["bash", "-c", cmd])
                if not success:
                    logger.warning(f"Failed to setup user: {output}")

        # Set VantaNode customization
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🎨 Setting up VantaNode...", ephemeral=True)
        else:
            await status_msg.edit(content="🎨 Setting up VantaNode...")
            
        # Create welcome message
        welcome_cmd = f"echo '{WELCOME_MESSAGE}' > /etc/motd"
        success, output = await run_docker_command(container_id, ["bash", "-c", welcome_cmd])
        if not success:
            logger.warning(f"Could not set welcome message: {output}")

        # Set hostname
        if not vps_id:
            vps_id = generate_vps_id()
        hostname_cmd = f"echo 'vantanode-{vps_id}' > /etc/hostname && hostname vantanode-{vps_id}"
        success, output = await run_docker_command(container_id, ["bash", "-c", hostname_cmd])
        if not success:
            logger.warning(f"Failed to set hostname: {output}")

        # Set watermark
        success, output = await run_docker_command(container_id, ["bash", "-c", f"echo '{WATERMARK}' > /etc/machine-info"])
        if not success:
            logger.warning(f"Could not set machine info: {output}")

        # Clean up
        cleanup_cmd = "apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*"
        await run_docker_command(container_id, ["bash", "-c", cleanup_cmd])

        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("✅ VantaNode VPS setup completed!", ephemeral=True)
        else:
            await status_msg.edit(content="✅ VantaNode VPS setup completed!")
            
        return True, ssh_password, vps_id
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
        logger.error(error_msg)
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send(f"❌ {error_msg}", ephemeral=True)
        else:
            await status_msg.edit(content=f"❌ {error_msg}")
        return False, None, None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = VantaNodeBot(command_prefix='/', intents=intents, help_command=None)

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    
    # Auto-start VPS containers based on status
    if bot.docker_client:
        for token, vps in bot.db.get_all_vps().items():
            if vps['status'] == 'running':
                try:
                    container = bot.docker_client.containers.get(vps["container_id"])
                    if container.status != "running":
                        container.start()
                        logger.info(f"Started container for VPS {vps['vps_id']}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {vps['container_id']} not found")
                except Exception as e:
                    logger.error(f"Error starting container: {e}")
    
    try:
        await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.streaming, name="📊VantaNode│VPS 🚀Prefix '/' ⚡"))
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} slash commands")
    except Exception as e:
        logger.error(f"Error syncing slash commands: {e}")

@bot.hybrid_command(name='ping', description='Ping the bot latency')
async def ping(ctx):
    try:
        latency = bot.latency * 1000
        await ctx.send(f"Pong Latency: {latency:.2f} ms")
    except Exception as e:
        logger.error(f"Error in ping command: {e}")
        await ctx.send("An error occurred.")

@bot.hybrid_command(name='help', description='Show all commands')
async def show_commands(ctx):
    try:
        embed = discord.Embed(title="🤖 VantaNode VPS Bot Commands", color=discord.Color.blue())
        
        embed.add_field(name="User Commands", value="""
`/deploy` - Create VPS (Admin)
`/list` - List your VPS
`/ping` - Check latency
`/help` - Show commands
`/manage_vps <vps_id>` - Manage VPS
`/transfer_vps <vps_id> <user>` - Transfer VPS
`/vps_stats <vps_id>` - Show usage
`/change_ssh_password <vps_id>` - Change SSH password
`/vps_shell <vps_id>` - Get shell access
`/vps_console <vps_id>` - Get console access
`/vps_usage` - Show usage stats
`/vps_uptime <vps_id>` - Show VPS uptime
""", inline=False)
        
        if has_admin_role(ctx):
            embed.add_field(name="Admin Commands", value="""
`/list_all` - List all VPS
`/delete_vps <vps_id>` - Delete VPS
`/admin_stats` - Show system stats
`/cleanup_vps` - Cleanup VPS
`/add_admin <user>` - Add admin
`/remove_admin <user>` - Remove admin
`/list_admins` - List admins
`/system_info` - System info
`/container_limit <max>` - Set limit
`/global_stats` - Global stats
`/edit_vps <vps_id>` - Edit VPS
`/ban_user <user>` - Ban user
`/unban_user <user>` - Unban user
`/list_banned` - List banned
`/backup_data` - Backup data
`/restore_data` - Restore data
`/resource_stats` - Show resource allocation
""", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in show_commands: {e}")
        await ctx.send("❌ An error occurred.")

@bot.hybrid_command(name='deploy', description='Create VPS (Admin only)')
@app_commands.describe(
    memory="Memory in GB",
    cpu="CPU cores",
    disk="Disk in GB",
    owner="Owner user",
    os_image="OS image",
    use_custom_image="Use custom image",
    reason="Reason (optional)"
)
async def create_vps_command(
    ctx,
    memory: int,
    cpu: int,
    disk: int,
    owner: discord.Member,
    os_image: str = DEFAULT_OS_IMAGE,
    use_custom_image: bool = True,
    reason: Optional[str] = None
):
    """Create a VPS with dedicated resources"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
        if bot.db.is_user_banned(owner.id):
            await ctx.send("❌ User banned!", ephemeral=True); return
        if not ctx.guild:
            await ctx.send("❌ Server only!", ephemeral=True); return

        try:
            if not bot.docker_client:
                bot.docker_client = docker.from_env()
            bot.docker_client.ping()
        except Exception:
            await ctx.send("❌ Docker not available.", ephemeral=True)
            return

        if not (1 <= memory <= 512):
            await ctx.send("❌ Memory 1-512GB", ephemeral=True); return
        if not (1 <= cpu <= 32):
            await ctx.send("❌ CPU 1-32 cores", ephemeral=True); return
        if not (10 <= disk <= 1000):
            await ctx.send("❌ Disk 10-1000GB", ephemeral=True); return

        containers = bot.docker_client.containers.list(all=True)
        if len(containers) >= bot.db.get_setting('max_containers', MAX_CONTAINERS):
            await ctx.send(f"❌ Max containers reached ({bot.db.get_setting('max_containers')}).", ephemeral=True)
            return
        if bot.db.get_user_vps_count(owner.id) >= bot.db.get_setting('max_vps_per_user', MAX_VPS_PER_USER):
            await ctx.send(f"❌ {owner.mention} has max VPS.", ephemeral=True)
            return

        status_msg = await ctx.send("🚀 Creating VantaNode VPS...")

        # Allocate CPU cores
        try:
            cpuset_cpus = allocate_cpu_cores(cpu)
        except Exception as e:
            await status_msg.edit(content=f"❌ {str(e)}")
            return

        vps_id = generate_vps_id()
        username = owner.name.lower().replace(" ", "_")[:20]
        root_password = generate_ssh_password()
        user_password = generate_ssh_password()
        token = generate_token()

        try:
            # Create Docker network if not exists
            nets = subprocess.run(
                ["docker","network","ls","--format","{{.Name}}"],
                capture_output=True, text=True
            ).stdout.splitlines()
            if DOCKER_NETWORK not in nets:
                subprocess.run(["docker","network","create", DOCKER_NETWORK], check=False)
        except Exception:
            pass

        try:
            # Build or pull image
            if use_custom_image:
                await status_msg.edit(content="🚀 Building VantaNode image...")
                image_tag = await build_custom_image(
                    vps_id, username, root_password, user_password, os_image
                )
            else:
                # Pull base image
                await status_msg.edit(content="📥 Pulling base image...")
                bot.docker_client.images.pull(os_image)
                image_tag = os_image
        except docker.errors.ImageNotFound:
            await status_msg.edit(content=f"❌ Image not found. Using default {DEFAULT_OS_IMAGE}")
            try:
                bot.docker_client.images.pull(DEFAULT_OS_IMAGE)
                image_tag = DEFAULT_OS_IMAGE
                os_image = DEFAULT_OS_IMAGE
            except Exception as e:
                await status_msg.edit(content=f"❌ Failed to pull default image: {e}")
                free_cpu_cores(cpuset_cpus)
                return
        except Exception as e:
            await status_msg.edit(content=f"❌ Failed to get image: {e}")
            free_cpu_cores(cpuset_cpus)
            return

        # Create container with dedicated resources
        await status_msg.edit(content="⚙️ Creating container with dedicated resources...")
        try:
            # Create volume for disk space
            volume_name = f"vantanode-{vps_id}-disk"
            try:
                bot.docker_client.volumes.create(
                    name=volume_name,
                    driver="local",
                    driver_opts={"type": "tmpfs", "device": "tmpfs", "o": f"size={disk}g"}
                )
            except docker.errors.APIError:
                # Volume might already exist
                pass

            # Create container with resource limits
            container = bot.docker_client.containers.run(
                image_tag,
                command="/usr/sbin/sshd -D" if use_custom_image else "/bin/bash -c 'while true; do sleep 86400; done'",
                detach=True,
                privileged=True,
                hostname=f"vantanode-{vps_id}",
                # Memory limits
                mem_limit=f"{memory}g",
                memswap_limit=f"{memory}g",  # Equal to memory = no swap
                mem_swappiness=0,
                # CPU limits
                cpuset_cpus=cpuset_cpus,
                cpu_period=100000,
                cpu_quota=int(cpu * 100000),
                cpu_shares=1024 * cpu,
                # Disk limits
                storage_opt={"size": f"{disk}G"},
                # Network
                network=DOCKER_NETWORK,
                # Mount disk volume
                volumes={volume_name: {'bind': '/', 'mode': 'rw'}},
                # Restart policy
                restart_policy={"Name": "always", "MaximumRetryCount": 3},
                name=f"vantanode-{vps_id}",
                # Resource isolation
                ulimits=[
                    docker.types.Ulimit(name='nofile', soft=65536, hard=65536),
                    docker.types.Ulimit(name='nproc', soft=65536, hard=65536),
                ]
            )
        except Exception as e:
            await status_msg.edit(content=f"❌ Failed to create container: {e}")
            free_cpu_cores(cpuset_cpus)
            return

        # Setup container
        await status_msg.edit(content="🔧 Setting up VPS...")
        setup_success, ssh_password, final_vps_id = await setup_container(
            container.id, status_msg, memory, username, vps_id=vps_id, use_custom_image=use_custom_image
        )
        
        if not setup_success:
            try:
                container.stop()
                container.remove(v=True)
            except Exception:
                pass
            free_cpu_cores(cpuset_cpus)
            await status_msg.edit(content="❌ Setup failed.")
            return

        # Get tmate session
        await status_msg.edit(content="🔐 Starting session...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container.id, "tmate", "-F",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            ssh_session_line = await capture_ssh_session_line(proc)
            if not ssh_session_line:
                raise Exception("Failed to get SSH session")
        except Exception as e:
            await status_msg.edit(content=f"❌ Failed to start session: {e}")
            try:
                container.stop()
                container.remove(v=True)
            except Exception:
                pass
            free_cpu_cores(cpuset_cpus)
            return

        # Save VPS data
        vps_data = {
            "token": token,
            "vps_id": final_vps_id,
            "container_id": container.id,
            "memory": memory,
            "cpu": cpu,
            "disk": disk,
            "username": username,
            "password": ssh_password,
            "root_password": root_password if use_custom_image else None,
            "created_by": str(owner.id),
            "created_at": datetime.datetime.now().isoformat(),
            "tmate_session": ssh_session_line,
            "watermark": WATERMARK,
            "os_image": os_image,
            "restart_count": 0,
            "last_restart": None,
            "status": "running",
            "use_custom_image": use_custom_image,
            "allocated_cpus": cpuset_cpus,
            "uptime_start": datetime.datetime.now().isoformat(),
            "total_uptime": 0
        }
        bot.db.add_vps(vps_data)
        
        # Update uptime tracker
        VPS_UPTIME_TRACKER[final_vps_id] = {
            'start_time': datetime.datetime.now(),
            'container_id': container.id
        }

        # Send success message
        try:
            embed = discord.Embed(title="🎉 VantaNode VPS Created", color=discord.Color.green())
            embed.add_field(name="VPS ID", value=final_vps_id, inline=True)
            embed.add_field(name="Memory", value=f"{memory}GB", inline=True)
            embed.add_field(name="CPU Cores", value=f"{cpu} cores", inline=True)
            embed.add_field(name="Disk", value=f"{disk}GB", inline=True)
            embed.add_field(name="CPU Isolation", value=cpuset_cpus, inline=True)
            if reason:
                embed.add_field(name="Reason", value=reason[:1024], inline=False)
            embed.add_field(name="Username", value=username, inline=True)
            embed.add_field(name="Password", value=f"||{ssh_password}||", inline=False)
            if use_custom_image:
                embed.add_field(name="Root Password", value=f"||{root_password}||", inline=False)
            embed.add_field(name="Session", value=f"```{ssh_session_line}```", inline=False)
            embed.add_field(name="SSH", value=f"```ssh {username}@<server>```", inline=False)

            await owner.send(embed=embed)
            await status_msg.edit(
                content=f"✅ VPS created for {owner.mention}. Check DMs."
            )
        except discord.Forbidden:
            await status_msg.edit(
                content=f"✅ VPS created for {owner.mention}. Enable DMs for credentials."
            )

    except Exception as e:
        logger.error(f"deploy error: {e}")
        await ctx.send(f"❌ Error: {e}")
        try:
            if 'container' in locals():
                try:
                    container.stop()
                    container.remove(v=True)
                except Exception:
                    pass
            if 'cpuset_cpus' in locals():
                free_cpu_cores(cpuset_cpus)
        except Exception:
            pass

@bot.hybrid_command(name='list', description='List your VPS')
async def list_vps(ctx):
    """List VPS instances owned by user"""
    try:
        user_vps = bot.db.get_user_vps(ctx.author.id)
        
        if not user_vps:
            await ctx.send("❌ No VPS found.", ephemeral=True)
            return

        embed = discord.Embed(title="Your VantaNode VPS", color=discord.Color.blue())
        
        for vps in user_vps:
            try:
                if vps["container_id"]:
                    container = bot.docker_client.containers.get(vps["container_id"])
                    status = container.status.capitalize()
                else:
                    status = "Unknown"
            except Exception:
                status = vps.get('status', 'Unknown').capitalize()

            # Calculate uptime
            uptime_seconds = vps.get('total_uptime', 0)
            if vps['vps_id'] in VPS_UPTIME_TRACKER:
                current_uptime = (datetime.datetime.now() - VPS_UPTIME_TRACKER[vps['vps_id']]['start_time']).total_seconds()
                uptime_seconds += int(current_uptime)
            
            uptime_str = str(datetime.timedelta(seconds=uptime_seconds))
            
            embed.add_field(
                name=f"VPS: {vps['vps_id']}",
                value=f"""
Status: {status}
Memory: {vps.get('memory', '?')}GB
CPU: {vps.get('cpu', '?')} cores
Disk: {vps.get('disk', '?')}GB
Uptime: {uptime_str}
Created: {vps.get('created_at', '?')[:10]}
""",
                inline=False
            )
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in list_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='list_all', description='List all VPS (Admin only)')
async def list_all_vps(ctx):
    """List all VPS instances (Admin only)"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        all_vps = bot.db.get_all_vps()
        
        if not all_vps:
            await ctx.send("❌ No VPS found.", ephemeral=True)
            return

        embed = discord.Embed(title="All VantaNode VPS", color=discord.Color.gold())
        
        for token, vps in list(all_vps.items())[:25]:  # Limit to 25 to avoid embed limits
            try:
                owner = await bot.fetch_user(int(vps['created_by']))
                owner_name = owner.name
            except:
                owner_name = f"User {vps['created_by']}"
                
            try:
                if vps["container_id"]:
                    container = bot.docker_client.containers.get(vps["container_id"])
                    status = container.status.capitalize()
                else:
                    status = "Unknown"
            except Exception:
                status = vps.get('status', 'Unknown').capitalize()

            embed.add_field(
                name=f"VPS: {vps['vps_id']}",
                value=f"""
Owner: {owner_name}
Status: {status}
Memory: {vps.get('memory', '?')}GB
CPU: {vps.get('cpu', '?')} cores
Disk: {vps.get('disk', '?')}GB
""",
                inline=True
            )
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in list_all_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='vps_stats', description='Show detailed stats for a VPS')
@app_commands.describe(vps_id="VPS ID")
async def vps_stats(ctx, vps_id: str):
    """Show detailed statistics for a VPS"""
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        # Check permissions
        caller_id = str(ctx.author.id)
        is_owner = caller_id == vps['created_by']
        is_admin = has_admin_role(ctx)
        
        if not (is_owner or is_admin):
            await ctx.send("❌ You don't have permission to view this VPS.", ephemeral=True)
            return

        # Get resource usage
        resource_usage = bot.db.get_resource_usage(vps_id)
        
        # Get container status
        container_status = "Unknown"
        try:
            if vps["container_id"]:
                container = bot.docker_client.containers.get(vps["container_id"])
                container_status = container.status.capitalize()
        except Exception:
            pass

        # Calculate uptime
        uptime_seconds = vps.get('total_uptime', 0)
        if vps_id in VPS_UPTIME_TRACKER:
            current_uptime = (datetime.datetime.now() - VPS_UPTIME_TRACKER[vps_id]['start_time']).total_seconds()
            uptime_seconds += int(current_uptime)
        
        uptime_str = str(datetime.timedelta(seconds=uptime_seconds))
        
        embed = discord.Embed(title=f"VPS Stats: {vps_id}", color=discord.Color.green())
        embed.add_field(name="Status", value=container_status, inline=True)
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Restarts", value=vps.get('restart_count', 0), inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=False)
        
        # Resource allocation
        embed.add_field(name="Allocated Resources", value="", inline=False)
        embed.add_field(name="Memory", value=f"{vps.get('memory', '?')}GB", inline=True)
        embed.add_field(name="CPU", value=f"{vps.get('cpu', '?')} cores", inline=True)
        embed.add_field(name="Disk", value=f"{vps.get('disk', '?')}GB", inline=True)
        
        if resource_usage:
            embed.add_field(name="\u200b", value="\u200b", inline=False)
            embed.add_field(name="Current Usage", value="", inline=False)
            embed.add_field(name="CPU Usage", value=f"{resource_usage.get('cpu_usage', 0):.1f}%", inline=True)
            embed.add_field(name="Memory Usage", value=f"{resource_usage.get('memory_usage', 0):.1f}%", inline=True)
            embed.add_field(name="Network TX", value=f"{resource_usage.get('network_tx', 0):.1f}MB", inline=True)
            embed.add_field(name="Network RX", value=f"{resource_usage.get('network_rx', 0):.1f}MB", inline=True)
            
            if resource_usage.get('last_updated'):
                last_updated = datetime.datetime.fromisoformat(resource_usage['last_updated'])
                embed.set_footer(text=f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in vps_stats: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='vps_uptime', description='Show VPS uptime')
@app_commands.describe(vps_id="VPS ID")
async def vps_uptime(ctx, vps_id: str):
    """Show detailed uptime information for a VPS"""
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        # Check permissions
        caller_id = str(ctx.author.id)
        is_owner = caller_id == vps['created_by']
        is_admin = has_admin_role(ctx)
        
        if not (is_owner or is_admin):
            await ctx.send("❌ You don't have permission to view this VPS.", ephemeral=True)
            return

        # Calculate total uptime
        total_uptime = vps.get('total_uptime', 0)
        current_uptime = 0
        
        if vps_id in VPS_UPTIME_TRACKER:
            current_uptime = (datetime.datetime.now() - VPS_UPTIME_TRACKER[vps_id]['start_time']).total_seconds()
        
        total_seconds = total_uptime + current_uptime
        
        # Format uptime
        days = int(total_seconds // (24 * 3600))
        hours = int((total_seconds % (24 * 3600)) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
        
        # Calculate uptime percentage if we have creation date
        created_at = datetime.datetime.fromisoformat(vps['created_at']) if 'created_at' in vps else None
        if created_at:
            total_lifetime = (datetime.datetime.now() - created_at).total_seconds()
            uptime_percentage = (total_seconds / total_lifetime) * 100 if total_lifetime > 0 else 100
        else:
            uptime_percentage = 100
        
        embed = discord.Embed(title=f"VPS Uptime: {vps_id}", color=discord.Color.blue())
        embed.add_field(name="Total Uptime", value=uptime_str, inline=False)
        embed.add_field(name="Uptime Percentage", value=f"{uptime_percentage:.2f}%", inline=True)
        embed.add_field(name="Restarts", value=vps.get('restart_count', 0), inline=True)
        
        if created_at:
            embed.add_field(name="Created", value=created_at.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        
        if vps.get('last_restart'):
            last_restart = datetime.datetime.fromisoformat(vps['last_restart'])
            embed.add_field(name="Last Restart", value=last_restart.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in vps_uptime: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='resource_stats', description='Show resource allocation stats (Admin only)')
async def resource_stats(ctx):
    """Show system resource allocation statistics"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        # Get system info
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get allocated resources
        allocated_memory = bot.db.get_setting('total_allocated_memory', 0)
        allocated_cpu = bot.db.get_setting('total_allocated_cpu', 0)
        allocated_disk = bot.db.get_setting('total_allocated_disk', 0)
        
        # Calculate percentages
        memory_percent = (allocated_memory / (mem.total / (1024 ** 3))) * 100 if mem.total > 0 else 0
        cpu_percent = (allocated_cpu / CPU_CORES_AVAILABLE) * 100 if CPU_CORES_AVAILABLE > 0 else 0
        disk_percent = (allocated_disk / (disk.total / (1024 ** 3))) * 100 if disk.total > 0 else 0
        
        embed = discord.Embed(title="System Resource Allocation", color=discord.Color.purple())
        
        # Memory stats
        embed.add_field(name="Memory", value=f"""
Total: {mem.total / (1024 ** 3):.1f} GB
Allocated: {allocated_memory} GB
Available: {(mem.total / (1024 ** 3)) - allocated_memory:.1f} GB
Usage: {memory_percent:.1f}%
""", inline=True)
        
        # CPU stats
        embed.add_field(name="CPU", value=f"""
Total Cores: {CPU_CORES_AVAILABLE}
Allocated Cores: {allocated_cpu}
Available Cores: {CPU_CORES_AVAILABLE - allocated_cpu}
Usage: {cpu_percent:.1f}%
""", inline=True)
        
        # Disk stats
        embed.add_field(name="Disk", value=f"""
Total: {disk.total / (1024 ** 3):.1f} GB
Allocated: {allocated_disk} GB
Available: {(disk.total / (1024 ** 3)) - allocated_disk:.1f} GB
Usage: {disk_percent:.1f}%
""", inline=True)
        
        # VPS count
        total_vps = len(bot.db.get_all_vps())
        running_vps = sum(1 for vps in bot.db.get_all_vps().values() if vps.get('status') == 'running')
        
        embed.add_field(name="VPS Statistics", value=f"""
Total VPS: {total_vps}
Running VPS: {running_vps}
Stopped VPS: {total_vps - running_vps}
Max Containers: {bot.db.get_setting('max_containers', MAX_CONTAINERS)}
""", inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in resource_stats: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='delete_vps', description='Delete a VPS (Admin only)')
@app_commands.describe(vps_id="VPS ID", reason="Reason for deletion (optional)")
async def delete_vps(ctx, vps_id: str, reason: Optional[str] = None):
    """Delete a VPS instance"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        # Free CPU cores
        if vps.get('allocated_cpus'):
            free_cpu_cores(vps['allocated_cpus'])
        
        # Remove from uptime tracker
        if vps_id in VPS_UPTIME_TRACKER:
            # Update uptime before removing
            current_time = datetime.datetime.now()
            uptime_seconds = (current_time - VPS_UPTIME_TRACKER[vps_id]['start_time']).total_seconds()
            bot.db.update_uptime(vps_id, int(uptime_seconds))
            del VPS_UPTIME_TRACKER[vps_id]
        
        # Stop and remove container
        try:
            if vps["container_id"] and bot.docker_client:
                container = bot.docker_client.containers.get(vps["container_id"])
                container.stop()
                container.remove(v=True)
                
                # Remove volume
                volume_name = f"vantanode-{vps_id}-disk"
                try:
                    volume = bot.docker_client.volumes.get(volume_name)
                    volume.remove()
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error removing container: {e}")
        
        # Remove from database
        bot.db.remove_vps(token)
        
        # Notify owner
        try:
            owner = await bot.fetch_user(int(vps['created_by']))
            embed = discord.Embed(title="VPS Deleted", color=discord.Color.red())
            embed.add_field(name="VPS ID", value=vps_id, inline=True)
            embed.add_field(name="Deleted By", value=ctx.author.name, inline=True)
            if reason:
                embed.add_field(name="Reason", value=reason, inline=False)
            embed.add_field(name="Resources Freed", value=f"""
Memory: {vps.get('memory', 0)}GB
CPU: {vps.get('cpu', 0)} cores
Disk: {vps.get('disk', 0)}GB
""", inline=False)
            await owner.send(embed=embed)
        except:
            pass
        
        await ctx.send(f"✅ VPS {vps_id} deleted successfully.")
        
    except Exception as e:
        logger.error(f"Error in delete_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='admin_stats', description='Show admin statistics (Admin only)')
async def admin_stats(ctx):
    """Show admin statistics"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        # Get system stats
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get Docker stats
        container_count = len(bot.docker_client.containers.list(all=True)) if bot.docker_client else 0
        running_containers = len(bot.docker_client.containers.list()) if bot.docker_client else 0
        
        # Get database stats
        total_vps = bot.db.get_stat('total_vps_created', 0)
        banned_users = len(bot.db.get_banned_users())
        admins = len(bot.db.get_admins())
        
        embed = discord.Embed(title="Admin Statistics", color=discord.Color.gold())
        
        embed.add_field(name="System Health", value=f"""
CPU Usage: {cpu_percent:.1f}%
Memory Usage: {mem.percent:.1f}%
Disk Usage: {disk.percent:.1f}%
""", inline=False)
        
        embed.add_field(name="Docker Status", value=f"""
Total Containers: {container_count}
Running Containers: {running_containers}
Max Allowed: {bot.db.get_setting('max_containers', MAX_CONTAINERS)}
""", inline=False)
        
        embed.add_field(name="VPS Statistics", value=f"""
Total VPS Created: {total_vps}
Current VPS Count: {len(bot.db.get_all_vps())}
Banned Users: {banned_users}
Admins: {admins}
""", inline=False)
        
        embed.add_field(name="Resource Allocation", value=f"""
Total Allocated Memory: {bot.db.get_setting('total_allocated_memory', 0)} GB
Total Allocated CPU: {bot.db.get_setting('total_allocated_cpu', 0)} cores
Total Allocated Disk: {bot.db.get_setting('total_allocated_disk', 0)} GB
""", inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in admin_stats: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='system_info', description='Show system information (Admin only)')
async def system_info(ctx):
    """Show detailed system information"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        # Get system information
        uname = platform.uname()
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        
        # Network info
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        # Docker info
        docker_version = "Unknown"
        docker_info = "Unknown"
        if bot.docker_client:
            try:
                docker_version = bot.docker_client.version()['Version']
                docker_info = bot.docker_client.info()
            except:
                pass
        
        embed = discord.Embed(title="System Information", color=discord.Color.dark_grey())
        
        embed.add_field(name="System", value=f"""
System: {uname.system}
Node Name: {uname.node}
Release: {uname.release}
Version: {uname.version}
Machine: {uname.machine}
Processor: {uname.processor}
""", inline=False)
        
        embed.add_field(name="Boot Time", value=boot_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
        embed.add_field(name="Hostname", value=hostname, inline=True)
        embed.add_field(name="IP Address", value=ip_address, inline=True)
        
        if docker_version != "Unknown":
            embed.add_field(name="Docker", value=f"""
Version: {docker_version}
Containers: {docker_info.get('Containers', 'N/A')}
Running: {docker_info.get('ContainersRunning', 'N/A')}
Paused: {docker_info.get('ContainersPaused', 'N/A')}
Stopped: {docker_info.get('ContainersStopped', 'N/A')}
""", inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in system_info: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='container_limit', description='Set maximum container limit (Admin only)')
@app_commands.describe(max_containers="Maximum number of containers allowed")
async def container_limit(ctx, max_containers: int):
    """Set maximum container limit"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        if max_containers < 1 or max_containers > 500:
            await ctx.send("❌ Limit must be between 1 and 500.", ephemeral=True)
            return

        bot.db.set_setting('max_containers', max_containers)
        await ctx.send(f"✅ Maximum container limit set to {max_containers}.")
        
    except Exception as e:
        logger.error(f"Error in container_limit: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='global_stats', description='Show global statistics')
async def global_stats(ctx):
    """Show global statistics"""
    try:
        # Get VPS statistics by user
        all_vps = bot.db.get_all_vps()
        user_vps_count = defaultdict(int)
        
        for vps in all_vps.values():
            user_vps_count[vps['created_by']] += 1
        
        top_users = sorted(user_vps_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate total resources
        total_memory = sum(vps.get('memory', 0) for vps in all_vps.values())
        total_cpu = sum(vps.get('cpu', 0) for vps in all_vps.values())
        total_disk = sum(vps.get('disk', 0) for vps in all_vps.values())
        
        # Calculate total uptime
        total_uptime = sum(vps.get('total_uptime', 0) for vps in all_vps.values())
        for vps_id, tracker in VPS_UPTIME_TRACKER.items():
            if vps_id in all_vps:
                current_uptime = (datetime.datetime.now() - tracker['start_time']).total_seconds()
                total_uptime += int(current_uptime)
        
        uptime_str = str(datetime.timedelta(seconds=total_uptime))
        
        embed = discord.Embed(title="Global VantaNode Statistics", color=discord.Color.blue())
        
        embed.add_field(name="Total Statistics", value=f"""
Total VPS: {len(all_vps)}
Total Memory Allocated: {total_memory} GB
Total CPU Cores Allocated: {total_cpu}
Total Disk Allocated: {total_disk} GB
Total Uptime: {uptime_str}
""", inline=False)
        
        if top_users:
            top_users_str = "\n".join([f"<@{user_id}>: {count} VPS" for user_id, count in top_users])
            embed.add_field(name="Top VPS Owners", value=top_users_str, inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in global_stats: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='edit_vps', description='Edit VPS resources (Admin only)')
@app_commands.describe(
    vps_id="VPS ID",
    memory="New memory in GB (optional)",
    cpu="New CPU cores (optional)",
    disk="New disk in GB (optional)"
)
async def edit_vps(ctx, vps_id: str, memory: Optional[int] = None, cpu: Optional[int] = None, disk: Optional[int] = None):
    """Edit VPS resources"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        updates = {}
        
        if memory is not None:
            if not (1 <= memory <= 512):
                await ctx.send("❌ Memory must be between 1-512GB", ephemeral=True)
                return
            updates['memory'] = memory
            
        if cpu is not None:
            if not (1 <= cpu <= 32):
                await ctx.send("❌ CPU must be between 1-32 cores", ephemeral=True)
                return
            
            # Free old CPU cores and allocate new ones
            if vps.get('allocated_cpus'):
                free_cpu_cores(vps['allocated_cpus'])
            
            try:
                new_cpuset = allocate_cpu_cores(cpu)
                updates['cpu'] = cpu
                updates['allocated_cpus'] = new_cpuset
            except Exception as e:
                await ctx.send(f"❌ {str(e)}", ephemeral=True)
                return
            
        if disk is not None:
            if not (10 <= disk <= 1000):
                await ctx.send("❌ Disk must be between 10-1000GB", ephemeral=True)
                return
            updates['disk'] = disk
        
        if not updates:
            await ctx.send("❌ No changes specified.", ephemeral=True)
            return
        
        # Update container if it exists
        if vps["container_id"] and bot.docker_client:
            try:
                container = bot.docker_client.containers.get(vps["container_id"])
                
                if memory is not None:
                    # Update memory limits
                    container.update(mem_limit=f"{memory}g", memswap_limit=f"{memory}g")
                
                if cpu is not None:
                    # Update CPU limits
                    container.update(
                        cpuset_cpus=updates.get('allocated_cpus', vps.get('allocated_cpus')),
                        cpu_period=100000,
                        cpu_quota=int(cpu * 100000),
                        cpu_shares=1024 * cpu
                    )
                
                if disk is not None:
                    # Note: Docker doesn't support dynamic disk resizing easily
                    # This would require creating a new volume
                    pass
                    
            except Exception as e:
                logger.warning(f"Error updating container: {e}")
        
        # Update database
        bot.db.update_vps(token, updates)
        
        # Update total allocated resources
        if memory is not None:
            total_mem = bot.db.get_setting('total_allocated_memory', 0) - vps.get('memory', 0) + memory
            bot.db.set_setting('total_allocated_memory', total_mem)
        
        if cpu is not None:
            total_cpu = bot.db.get_setting('total_allocated_cpu', 0) - vps.get('cpu', 0) + cpu
            bot.db.set_setting('total_allocated_cpu', total_cpu)
        
        if disk is not None:
            total_disk = bot.db.get_setting('total_allocated_disk', 0) - vps.get('disk', 0) + disk
            bot.db.set_setting('total_allocated_disk', total_disk)
        
        await ctx.send(f"✅ VPS {vps_id} updated successfully.")
        
    except Exception as e:
        logger.error(f"Error in edit_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='manage_vps', description='Manage your VPS (start/stop/restart/status)')
@app_commands.describe(vps_id="VPS ID", action="Action (start/stop/restart/status)")
async def manage_vps(ctx, vps_id: str, action: str):
    """Manage VPS (start/stop/restart/status)"""
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        # Check permissions
        caller_id = str(ctx.author.id)
        is_owner = caller_id == vps['created_by']
        is_admin = has_admin_role(ctx)
        
        if not (is_owner or is_admin):
            await ctx.send("❌ You don't have permission to manage this VPS.", ephemeral=True)
            return

        if not bot.docker_client:
            await ctx.send("❌ Docker not available.", ephemeral=True)
            return

        try:
            container = bot.docker_client.containers.get(vps["container_id"])
        except docker.errors.NotFound:
            await ctx.send("❌ Container not found.", ephemeral=True)
            return

        action = action.lower()
        
        if action == 'status':
            status = container.status.capitalize()
            embed = discord.Embed(title=f"VPS Status: {vps_id}", color=discord.Color.blue())
            embed.add_field(name="Status", value=status, inline=True)
            embed.add_field(name="Created", value=container.attrs['Created'][:19], inline=True)
            await ctx.send(embed=embed)
            
        elif action == 'start':
            if container.status == 'running':
                await ctx.send("❌ VPS is already running.", ephemeral=True)
                return
                
            container.start()
            
            # Update uptime tracker
            VPS_UPTIME_TRACKER[vps_id] = {
                'start_time': datetime.datetime.now(),
                'container_id': container.id
            }
            
            bot.db.update_vps(token, {'status': 'running'})
            await ctx.send(f"✅ VPS {vps_id} started successfully.")
            
        elif action == 'stop':
            if container.status != 'running':
                await ctx.send("❌ VPS is not running.", ephemeral=True)
                return
                
            container.stop()
            
            # Update uptime before stopping
            if vps_id in VPS_UPTIME_TRACKER:
                current_time = datetime.datetime.now()
                uptime_seconds = (current_time - VPS_UPTIME_TRACKER[vps_id]['start_time']).total_seconds()
                bot.db.update_uptime(vps_id, int(uptime_seconds))
                del VPS_UPTIME_TRACKER[vps_id]
            
            bot.db.update_vps(token, {'status': 'stopped'})
            await ctx.send(f"✅ VPS {vps_id} stopped successfully.")
            
        elif action == 'restart':
            container.restart()
            
            # Update restart count
            restart_count = vps.get('restart_count', 0) + 1
            bot.db.update_vps(token, {
                'restart_count': restart_count,
                'last_restart': datetime.datetime.now().isoformat()
            })
            
            # Reset uptime tracker
            VPS_UPTIME_TRACKER[vps_id] = {
                'start_time': datetime.datetime.now(),
                'container_id': container.id
            }
            
            await ctx.send(f"✅ VPS {vps_id} restarted successfully.")
            
        else:
            await ctx.send("❌ Invalid action. Use: start, stop, restart, status", ephemeral=True)
            
    except Exception as e:
        logger.error(f"Error in manage_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='transfer_vps', description='Transfer VPS ownership (Admin only)')
@app_commands.describe(vps_id="VPS ID", new_owner="New owner user")
async def transfer_vps(ctx, vps_id: str, new_owner: discord.Member):
    """Transfer VPS ownership"""
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return

        if bot.db.is_user_banned(new_owner.id):
            await ctx.send("❌ New owner is banned!", ephemeral=True)
            return

        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found.", ephemeral=True)
            return

        # Check if new owner has reached limit
        if bot.db.get_user_vps_count(new_owner.id) >= bot.db.get_setting('max_vps_per_user', MAX_VPS_PER_USER):
            await ctx.send(f"❌ {new_owner.mention} has reached the maximum VPS limit.", ephemeral=True)
            return

        old_owner_id = vps['created_by']
        
        # Update database
        bot.db.update_vps(token, {
            'created_by': str(new_owner.id),
            'username': new_owner.name.lower().replace(" ", "_")[:20]
        })
        
        # Notify both users
        try:
            old_owner = await bot.fetch_user(int(old_owner_id))
            embed = discord.Embed(title="VPS Transferred", color=discord.Color.orange())
            embed.add_field(name="VPS ID", value=vps_id, inline=True)
            embed.add_field(name="Transferred To", value=new_owner.name, inline=True)
            embed.add_field(name="Transferred By", value=ctx.author.name, inline=True)
            await old_owner.send(embed=embed)
        except:
            pass
        
        try:
            embed = discord.Embed(title="VPS Received", color=discord.Color.green())
            embed.add_field(name="VPS ID", value=vps_id, inline=True)
            embed.add_field(name="Memory", value=f"{vps.get('memory', '?')}GB", inline=True)
            embed.add_field(name="CPU", value=f"{vps.get('cpu', '?')} cores", inline=True)
            embed.add_field(name="Disk", value=f"{vps.get('disk', '?')}GB", inline=True)
            embed.add_field(name="Transferred From", value=f"User {old_owner_id}", inline=True)
            await new_owner.send(embed=embed)
        except:
            pass
        
        await ctx.send(f"✅ VPS {vps_id} transferred from <@{old_owner_id}> to {new_owner.mention}.")
        
    except Exception as e:
        logger.error(f"Error in transfer_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

# Additional basic commands
@bot.hybrid_command(name='ban_user', description='Ban a user from using the bot (Admin only)')
@app_commands.describe(user="User to ban", reason="Reason for ban (optional)")
async def ban_user(ctx, user: discord.Member, reason: Optional[str] = None):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        bot.db.ban_user(user.id)
        embed = discord.Embed(title="User Banned", color=discord.Color.red())
        embed.add_field(name="User", value=user.mention, inline=True)
        embed.add_field(name="Banned By", value=ctx.author.mention, inline=True)
        if reason:
            embed.add_field(name="Reason", value=reason, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in ban_user: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='unban_user', description='Unban a user (Admin only)')
@app_commands.describe(user="User to unban")
async def unban_user(ctx, user: discord.Member):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        bot.db.unban_user(user.id)
        await ctx.send(f"✅ {user.mention} has been unbanned.")
    except Exception as e:
        logger.error(f"Error in unban_user: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='list_banned', description='List all banned users (Admin only)')
async def list_banned(ctx):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        banned_users = bot.db.get_banned_users()
        if not banned_users:
            await ctx.send("✅ No users are banned.")
            return
            
        embed = discord.Embed(title="Banned Users", color=discord.Color.red())
        for user_id in banned_users:
            try:
                user = await bot.fetch_user(int(user_id))
                embed.add_field(name="User", value=f"{user.name} ({user.id})", inline=False)
            except:
                embed.add_field(name="User", value=f"Unknown ({user_id})", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in list_banned: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='add_admin', description='Add an admin (Admin only)')
@app_commands.describe(user="User to make admin")
async def add_admin(ctx, user: discord.Member):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        bot.db.add_admin(user.id)
        await ctx.send(f"✅ {user.mention} has been added as an admin.")
    except Exception as e:
        logger.error(f"Error in add_admin: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='remove_admin', description='Remove an admin (Admin only)')
@app_commands.describe(user="User to remove admin")
async def remove_admin(ctx, user: discord.Member):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        # Don't allow removing yourself if you're the only admin
        admins = bot.db.get_admins()
        if len(admins) <= 1 and str(user.id) in admins:
            await ctx.send("❌ Cannot remove the only admin.")
            return
            
        bot.db.remove_admin(user.id)
        await ctx.send(f"✅ {user.mention} has been removed as an admin.")
    except Exception as e:
        logger.error(f"Error in remove_admin: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='list_admins', description='List all admins')
async def list_admins(ctx):
    try:
        admins = bot.db.get_admins()
        if not admins:
            await ctx.send("✅ No admins found.")
            return
            
        embed = discord.Embed(title="Admins", color=discord.Color.green())
        for user_id in admins:
            try:
                user = await bot.fetch_user(int(user_id))
                embed.add_field(name="Admin", value=f"{user.name} ({user.id})", inline=False)
            except:
                embed.add_field(name="Admin", value=f"Unknown ({user_id})", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in list_admins: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='backup_data', description='Backup all bot data (Admin only)')
async def backup_data(ctx):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        if bot.db.backup_data():
            await ctx.send("✅ Data backup completed successfully.")
        else:
            await ctx.send("❌ Failed to backup data.")
    except Exception as e:
        logger.error(f"Error in backup_data: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='restore_data', description='Restore data from backup (Admin only)')
async def restore_data(ctx):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        # Warning message
        await ctx.send("⚠️ **WARNING:** This will overwrite all current data. Type `CONFIRM` to proceed.")
        
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content == "CONFIRM"
        
        try:
            await bot.wait_for('message', timeout=30.0, check=check)
        except asyncio.TimeoutError:
            await ctx.send("❌ Restoration cancelled (timeout).")
            return
        
        if bot.db.restore_data():
            await ctx.send("✅ Data restoration completed successfully.")
        else:
            await ctx.send("❌ Failed to restore data.")
    except Exception as e:
        logger.error(f"Error in restore_data: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

@bot.hybrid_command(name='cleanup_vps', description='Cleanup unused VPS (Admin only)')
async def cleanup_vps(ctx):
    try:
        if not has_admin_role(ctx):
            await ctx.send("❌ Admin only!", ephemeral=True)
            return
            
        await ctx.send("🔄 Cleaning up unused VPS...")
        
        all_vps = bot.db.get_all_vps()
        cleaned = 0
        
        for token, vps in all_vps.items():
            try:
                if not vps["container_id"]:
                    continue
                    
                container = bot.docker_client.containers.get(vps["container_id"])
                if container.status != "running":
                    # Container exists but isn't running
                    container.remove(v=True)
                    bot.db.remove_vps(token)
                    cleaned += 1
                    
            except docker.errors.NotFound:
                # Container doesn't exist
                bot.db.remove_vps(token)
                cleaned += 1
            except Exception as e:
                logger.error(f"Error cleaning up VPS {vps['vps_id']}: {e}")
        
        await ctx.send(f"✅ Cleanup completed. Removed {cleaned} unused VPS.")
        
    except Exception as e:
        logger.error(f"Error in cleanup_vps: {e}")
        await ctx.send(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    try:
        os.makedirs("temp_dockerfiles", exist_ok=True)
        os.makedirs("migrations", exist_ok=True)
        
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        traceback.print_exc()
# =========================================== BOT CODE ENDS HERE ===========================================
