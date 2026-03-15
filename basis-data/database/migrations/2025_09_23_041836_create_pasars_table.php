<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        Schema::create('pasars', function (Blueprint $table) {
            $table->id('id_pasar');
            $table->string('nama_pasar');
            $table->foreignId('id_user')->constrained('users', 'id_user')->onDelete('cascade');
            $table->timestamps();
        });
    }
    public function down(): void { Schema::dropIfExists('pasars'); }
};