<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('top_ups', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained('users', 'id_user');
            $table->decimal('amount', 15, 2);
            $table->string('reference_code')->unique();
            $table->enum('status', ['pending', 'completed', 'failed'])->default('pending');
            $table->enum('payment_method', ['bank', 'mini_market'])->default('bank');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('top_ups');
    }
};