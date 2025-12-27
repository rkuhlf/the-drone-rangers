"""
Drone Management API

This module handles the creation, listing, and deletion of drones in the simulation.
It maintains a persistent store of drone configurations (make/model) and synchronizes
the simulation state when drones are added or removed.
"""

import os
import pathlib
import re
import sqlite3


import numpy as np
from flask import Blueprint, jsonify, request

from simulation import world

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

TMP_DIRECTORY = os.path.join(os.path.dirname(__file__), "tmp")
pathlib.Path(TMP_DIRECTORY).mkdir(exist_ok=True)
DB_PATH = os.path.join(TMP_DIRECTORY, "drones.sqlite3")


# -----------------------------------------------------------------------------
# Database Helpers
# -----------------------------------------------------------------------------


def _get_db_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _generate_next_drone_id(conn: sqlite3.Connection) -> str:
    """Generate the next available drone ID (e.g., DR-001)."""
    cur = conn.execute("SELECT id FROM drones")
    max_num = 0
    for row in cur.fetchall():
        m = re.match(r"^DR-(\d{3,})$", row["id"])  # allow 3+ digits
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"DR-{(max_num + 1):03d}"


# -----------------------------------------------------------------------------
# Blueprint Factory
# -----------------------------------------------------------------------------


def create_drones_blueprint(W: world.World) -> Blueprint:
    """
    Factory to create a drones blueprint with a dependency on the World instance.

    Args:
        W: The World instance to synchronize drone state with.
    """

    # Initialize database
    conn = _get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drones (
                id TEXT PRIMARY KEY,
                make TEXT NOT NULL,
                model TEXT NOT NULL
            )
            """
        )
        conn.commit()

        # Sync world state with database: initialize drone positions
        rows = conn.execute("SELECT id FROM drones").fetchall()
        # Random positions for existing drones
        W.drones = np.random.rand(len(rows), 2) * 5
    finally:
        conn.close()

    drones_bp = Blueprint("drones", __name__)

    @drones_bp.route("/drones", methods=["GET"])
    def list_drones():
        """List all registered drones."""
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                "SELECT id, make, model FROM drones ORDER BY id"
            ).fetchall()
            items = [dict(row) for row in rows]
            return jsonify({"items": items, "total": len(items)}), 200
        finally:
            conn.close()

    @drones_bp.route("/drones", methods=["POST"])
    def create_drone():
        """Create a new drone."""
        data = request.get_json(silent=True) or {}
        make = (data.get("make") or "").strip()
        model = (data.get("model") or "").strip()

        if not make or not model:
            return jsonify({"error": "'make' and 'model' are required"}), 400

        conn = _get_db_connection()
        try:
            drone_id = _generate_next_drone_id(conn)
            conn.execute(
                "INSERT INTO drones (id, make, model) VALUES (?, ?, ?)",
                (drone_id, make, model),
            )
            conn.commit()

            # Sync with simulation: Add a new drone
            # TODO: Do something a little bit smarter with tracking the ID drone-by-drone.
            # Currently just appending a random position
            W.drones = np.concatenate([W.drones, np.random.randint(0, 6, size=(1, 2))])

            return jsonify({"id": drone_id, "make": make, "model": model}), 201
        finally:
            conn.close()

    @drones_bp.route("/drones/<drone_id>", methods=["DELETE"])
    def delete_drone(drone_id: str):
        """Delete a drone by ID."""
        conn = _get_db_connection()
        try:
            cur = conn.execute("DELETE FROM drones WHERE id = ?", (drone_id,))
            conn.commit()
            if cur.rowcount == 0:
                return jsonify({"error": "drone not found"}), 404

            # Sync with simulation: Remove the last drone
            # TODO: Do something a little bit smarter with tracking the ID drone-by-drone.
            # Currently just removing the last one
            if len(W.drones) > 0:
                W.drones = W.drones[:-1]

            return ("", 204)
        finally:
            conn.close()

    return drones_bp
